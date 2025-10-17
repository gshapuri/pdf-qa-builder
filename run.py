#!/usr/bin/env python3

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path
from typing import List

from pdf_pipeline.pdf_reader import PDFReader
from pdf_pipeline.text_splitter import TextSplitter
from pdf_pipeline.embedder import Embedder
from pdf_pipeline.summarizer import Summarizer
from pdf_pipeline.vector_store import WeaviateVectorStore
from query_pipeline.retriever import Retriever
from query_pipeline.generator import Generator


def build_index(
    pdf_dir: str, 
    chunk_size: int = 250, 
    overlap: int = 50,
    summary_level: str = "section"
):
    """
    read pdfs from folder, chunk, embed both summaries and chunks
    
    :param pdf_dir: directory path containing pdf files
    :param chunk_size: number of words per chunk
    :param overlap: number of words overlapping between chunks
    :param summary_level: 'section' or 'chunk' - what to summarize
    """
    start_time = time.time()
    
    pdf_files: List[str] = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    print(f"Summary level: {summary_level}")
    
    # initialize pipeline components
    reader = PDFReader()
    splitter = TextSplitter(chunk_size=chunk_size, overlap=overlap)
    embedder = Embedder(model_name="balanced")
    summarizer = Summarizer(model_name="fast")
    store = WeaviateVectorStore()
    
    processed_count = 0
    
    for pdf_path in pdf_files:
        pdf_name = Path(pdf_path).name
        print(f"\nProcessing {pdf_name}...")
        
        # extract sections with error handling
        try:
            sections = reader.extract_sections(pdf_path)
        except Exception as e:
            print(f"Failed to process {pdf_name}: {e}")
            continue
        
        # skip if no sections extracted
        if not sections:
            print(f"Skipping {pdf_name}: no extractable text")
            continue
        
        # split into chunks
        chunks = splitter.split_sections(sections)
        print(f"Created {len(chunks)} chunks from {len(sections)} sections")
        
        # generate summaries based on level
        if summary_level == "section":
            print(f"Generating section summaries...")
            summaries = summarizer.summarize_sections(sections)
            
            # embed section summaries
            print(f"Creating embeddings for summaries...")
            summary_texts = list(summaries.values())
            summary_embeds = embedder.embed_texts(summary_texts)
            
            summary_vectors = {
                name: vector 
                for name, vector in zip(summaries.keys(), summary_embeds)
            }
            
        else:  # chunk level
            print(f"Generating chunk summaries...")
            chunk_summaries = summarizer.summarize_chunks(chunks)
            
            # embed chunk summaries
            print(f"Creating embeddings for summaries...")
            summary_texts = [cs["summary"] for cs in chunk_summaries]
            summary_embeds = embedder.embed_texts(summary_texts)
            
            # organize by section for storage
            summaries = {}
            summary_vectors = {}
            for chunk, vector in zip(chunk_summaries, summary_embeds):
                key = f"{chunk['section']}_chunk_{chunk['chunk_id']}"
                summaries[key] = chunk["summary"]
                summary_vectors[key] = vector
        
        # embed chunks
        print(f"Creating embeddings for chunks...")
        chunk_texts = [chunk.text for chunk in chunks]
        chunk_vectors = embedder.embed_texts(chunk_texts)
        
        # prepare chunks for storage
        chunk_dicts = [
            {
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
            }
            for chunk in chunks
        ]
        
        # upload to weaviate
        store.insert_sections(
            sections=sections,
            summaries=summaries,
            summary_vectors=summary_vectors,
            chunks=chunk_dicts,
            chunk_vectors=chunk_vectors,
            pdf_name=pdf_name,
            summary_level=summary_level,
        )
        
        processed_count += 1
    
    elapsed_time = time.time() - start_time
    print(f"\nAll PDFs processed successfully: {processed_count}/{len(pdf_files)} in {elapsed_time:.1f}s")


def ask_question(question: str, top_k: int = 5):
    """
    query weaviate and generate answer using llm
    
    :param question: natural language query
    :param top_k: number of sections to retrieve
    """
    # initialize pipeline components
    retriever = Retriever(top_k=top_k)
    generator = Generator()

    try:
        retrieved_results = retriever.retrieve(question)
        
        if not retrieved_results:
            print("No results found")
            return
        
        result = generator.generate(question, retrieved_results)
        
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}\n")
        print(f"{result['answer']}\n")
        print(f"{'='*80}")
        print(f"Sources:")
        print(f"{'='*80}")
        for i, source in enumerate(result['sources'], 1):
            print(f"{i}. {source['pdf']} - {source['section']} ({source['num_chunks']} chunks)")
        print()
    
    finally:
        retriever.close()
        generator.close()
        

def main():
    """
    main entry point for pdf qa builder cli
    """
    parser = argparse.ArgumentParser(
        description="PDF QA Builder CLI — Build index or ask questions"
    )
    subparsers = parser.add_subparsers(dest="command")
    
    # index command
    index_parser = subparsers.add_parser("index", help="Read PDFs and build index")
    index_parser.add_argument(
        "--pdf-dir", 
        required=True, 
        help="Folder path containing PDF files"
    )
    index_parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=250, 
        help="Number of words per chunk"
    )
    index_parser.add_argument(
        "--overlap", 
        type=int, 
        default=50, 
        help="Number of words overlap"
    )
    index_parser.add_argument(
        "--summary-level",
        type=str,
        choices=["section", "chunk"],
        default="section",
        help="Summarization level: section (one summary per section, default) or chunk (one summary per chunk)"
    )
    
    # ask command
    ask_parser = subparsers.add_parser("ask", help="Query the Weaviate index")
    ask_parser.add_argument(
        "question", 
        type=str, 
        help="Question to ask"
    )
    ask_parser.add_argument(
        "--top-k", 
        type=int, 
        default=5, 
        help="Number of sections to return"
    )
    
    args = parser.parse_args()
    
    if args.command == "index":
        build_index(
            args.pdf_dir, 
            args.chunk_size, 
            args.overlap,
            args.summary_level
        )
    elif args.command == "ask":
        ask_question(args.question, args.top_k)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
