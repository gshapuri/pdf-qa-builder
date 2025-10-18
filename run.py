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
from query_pipeline.reranker import Reranker
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
    
    try:
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
            
            try:
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
                
            except Exception as e:
                print(f"Error processing {pdf_name}: {e}")
                continue
        
        elapsed_time = time.time() - start_time
        print(f"\nAll PDFs processed successfully: {processed_count}/{len(pdf_files)} in {elapsed_time:.1f}s")
        
    except Exception as e:
        print(f"Error during indexing: {e}")
        raise
    finally:
        if 'store' in locals():
            store.close()


def ask_question(
    question: str, 
    top_k: int = 5,
    rerank_mode: str = "hybrid",
    rerank_model: str = "balanced",
    retrieve_k: int = 20,
):
    """
    query weaviate and generate answer using llm
    
    :param question: natural language query
    :param top_k: number of chunks/sections to use for answer generation
    :param rerank_mode: 'chunks' (flat), 'sections' (per section), 'hybrid' (balanced), or 'none' (no reranking)
    :param rerank_model: reranker model ('fast', 'balanced', 'best', 'llm')
    :param retrieve_k: number of initial candidates to retrieve (before reranking)
    """
    use_reranker = rerank_mode != "none"
    
    try:
        # initialize pipeline components
        retriever = Retriever(top_k=retrieve_k if use_reranker else top_k)
        generator = Generator()
        
        if use_reranker:
            reranker = Reranker(model_name=rerank_model)

        try:
            # retrieve candidates
            retrieved_results = retriever.retrieve(question)
            
            if not retrieved_results:
                print("No results found")
                return
            
            # rerank if enabled
            if use_reranker:
                print(f"\nReranking with mode: {rerank_mode}")
                
                try:
                    if rerank_mode == "chunks":
                        # flat reranking: get top k chunks globally
                        reranked_chunks = reranker.rerank_chunks(
                            question, 
                            retrieved_results, 
                            top_k=top_k
                        )
                        
                        # reorganize into section format for generator
                        reranked_results = []
                        section_map = {}
                        
                        for chunk in reranked_chunks:
                            key = (chunk['section'], chunk['pdf_name'])
                            if key not in section_map:
                                section_map[key] = {
                                    'section': chunk['section'],
                                    'pdf_name': chunk['pdf_name'],
                                    'chunks': []
                                }
                            section_map[key]['chunks'].append(chunk)
                        
                        reranked_results = list(section_map.values())
                        
                    elif rerank_mode == "sections":
                        # per-section reranking: keep top chunks per section
                        chunks_per_section = max(1, top_k // len(retrieved_results))
                        reranked_results = reranker.rerank_sections(
                            question,
                            retrieved_results,
                            chunks_per_section=chunks_per_section,
                        )
                        
                    # hybrid    
                    else:
                        reranked_results = reranker.rerank_hybrid(
                            question,
                            retrieved_results,
                            total_chunks=top_k,
                            min_chunks_per_section=1,
                            max_chunks_per_section=3,
                        )
                    
                    # show reranking stats
                    total_chunks = sum(len(r['chunks']) for r in reranked_results)
                    print(f"After reranking: {len(reranked_results)} sections, {total_chunks} chunks")
                    
                    final_results = reranked_results
                    
                except Exception as e:
                    print(f"Error during reranking: {e}")
                    print("Falling back to retrieved results without reranking")
                    final_results = retrieved_results
            else:
                final_results = retrieved_results
            
            # generate answer
            result = generator.generate(question, final_results)
            
            print(f"\n{'='*80}")
            print(f"Question: {question}")
            print(f"{'='*80}\n")
            print(f"{result['answer']}\n")
            print(f"{'='*80}")
            print(f"Sources:")
            print(f"{'='*80}")
            for i, source in enumerate(result['sources'], 1):
                print(f"{i}. {source['pdf']} - {source['section']} ({source['num_chunks']} chunks)")
            
            # show relevance scores if available
            if use_reranker and rerank_mode != "sections":
                print(f"\n{'='*80}")
                print(f"Relevance Scores:")
                print(f"{'='*80}")
                for section_result in final_results:
                    section = section_result['section']
                    pdf = section_result['pdf_name']
                    for chunk in section_result['chunks']:
                        score = chunk.get('relevance_score', 0.0)
                        print(f"  [{pdf}] {section} (chunk {chunk['chunk_id']}): {score:.4f}")
            
            print()
            
        except Exception as e:
            print(f"Error during query execution: {e}")
            raise
        
    except Exception as e:
        print(f"Error initializing query pipeline: {e}")
        raise
    finally:
        if 'retriever' in locals():
            retriever.close()
        if 'generator' in locals():
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
        default=10, 
        help="Number of chunks to use for answer generation"
    )
    ask_parser.add_argument(
        "--rerank-mode",
        type=str,
        choices=["chunks", "sections", "hybrid", "none"],
        default="hybrid",
        help="Reranking strategy: chunks (flat), sections (per-section), hybrid (balanced), none (no reranking)"
    )
    ask_parser.add_argument(
        "--rerank-model",
        type=str,
        choices=["fast", "balanced", "best", "llm"],
        default="balanced",
        help="Reranker model: fast (MiniLM-L6), balanced (MiniLM-L12, default), best (BGE), llm (GPT-4o-mini)"
    )
    ask_parser.add_argument(
        "--retrieve-k",
        type=int,
        default=20,
        help="Number of initial candidates to retrieve before reranking"
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
        ask_question(
            args.question, 
            args.top_k,
            rerank_mode=args.rerank_mode,
            rerank_model=args.rerank_model,
            retrieve_k=args.retrieve_k,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
