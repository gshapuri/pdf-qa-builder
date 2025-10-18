from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from sentence_transformers import CrossEncoder


# predefined reranker models
DEFAULT_RERANKERS = {
    "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "balanced": "cross-encoder/ms-marco-MiniLM-L-12-v2", 
    "best": "BAAI/bge-reranker-base",
}


class Reranker:
    """
    rerank retrieved chunks using a cross-encoder model
    
    cross-encoders are much more accurate than bi-encoders (embeddings)
    because they see both query and document together, but slower
    
    typical pipeline:
    1. retriever uses fast bi-encoder to get top 50-100 candidates
    2. reranker uses slower cross-encoder to find best 5-10 chunks
    """

    def __init__(
        self, 
        model_name: str = "balanced",
        batch_size: int = 16,
        device: str = "cpu"
    ):
        """
        initialize reranker with a cross-encoder model
        
        :param model_name: model preset ('fast', 'balanced', 'best') or full model path
        :param batch_size: number of pairs to score per batch
        :param device: device to use ('cpu' or 'cuda')
        """
        # resolve preset names to actual model paths
        if model_name in DEFAULT_RERANKERS:
            resolved_model = DEFAULT_RERANKERS[model_name]
        else:
            resolved_model = model_name
        
        self.model_name = resolved_model
        self.batch_size = batch_size
        self.device = device
        
        print(f"Loading reranker model: {resolved_model}")
        try:
            self.model = CrossEncoder(resolved_model, device=device)
        except Exception as e:
            print(f"Failed to load {resolved_model}: {e}")
            print("Falling back to fast model...")
            self.model_name = DEFAULT_RERANKERS["fast"]
            self.model = CrossEncoder(self.model_name, device=device)

    def rerank_chunks(
        self,
        question: str,
        retrieved_results: List[Dict[str, Any]],
        top_k: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        rerank all chunks across all sections and return top k
        
        :param question: user query
        :param retrieved_results: results from retriever (list of sections with chunks)
        :param top_k: number of chunks to return
        :param min_score: optional minimum relevance score threshold
        :return: list of top-k chunks with scores
        """
        if not retrieved_results:
            return []
        
        # flatten all chunks from all sections
        all_chunks = []
        for result in retrieved_results:
            section = result.get('section', 'Unknown')
            pdf_name = result.get('pdf_name', 'Unknown')
            chunks = result.get('chunks', [])
            
            for chunk in chunks:
                all_chunks.append({
                    'text': chunk['text'],
                    'chunk_id': chunk['chunk_id'],
                    'section': section,
                    'pdf_name': pdf_name,
                    'is_match': chunk.get('is_match', False),
                })
        
        if not all_chunks:
            return []
        
        print(f"Reranking {len(all_chunks)} chunks...")
        
        # create query-document pairs
        pairs = [[question, chunk['text']] for chunk in all_chunks]
        
        # score all pairs
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=True)
        
        # attach scores to chunks
        for chunk, score in zip(all_chunks, scores):
            chunk['relevance_score'] = float(score)
        
        # filter by minimum score if specified
        if min_score is not None:
            all_chunks = [c for c in all_chunks if c['relevance_score'] >= min_score]
        
        # sort by relevance score (descending)
        all_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # return top k
        top_chunks = all_chunks[:top_k]
        
        print(f"Top {len(top_chunks)} chunks selected (scores: {top_chunks[0]['relevance_score']:.3f} to {top_chunks[-1]['relevance_score']:.3f})")
        
        return top_chunks

    def rerank_sections(
        self,
        question: str,
        retrieved_results: List[Dict[str, Any]],
        chunks_per_section: int = 3,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        rerank chunks within each section and keep top chunks per section
        preserves section structure but filters to most relevant chunks
        
        :param question: user query
        :param retrieved_results: results from retriever
        :param chunks_per_section: number of top chunks to keep per section
        :param min_score: optional minimum relevance score threshold
        :return: list of sections with reranked chunks
        """
        if not retrieved_results:
            return []
        
        reranked_results = []
        
        for result in retrieved_results:
            section = result.get('section', 'Unknown')
            pdf_name = result.get('pdf_name', 'Unknown')
            chunks = result.get('chunks', [])
            
            if not chunks:
                continue
            
            print(f"Reranking {len(chunks)} chunks in section: {section}")
            
            # create query-document pairs for this section
            pairs = [[question, chunk['text']] for chunk in chunks]
            
            # score all chunks in this section
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            
            # attach scores
            scored_chunks = []
            for chunk, score in zip(chunks, scores):
                scored_chunks.append({
                    'text': chunk['text'],
                    'chunk_id': chunk['chunk_id'],
                    'is_match': chunk.get('is_match', False),
                    'relevance_score': float(score),
                })
            
            # filter by minimum score if specified
            if min_score is not None:
                scored_chunks = [c for c in scored_chunks if c['relevance_score'] >= min_score]
            
            # sort by relevance score
            scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # keep top chunks per section
            top_chunks = scored_chunks[:chunks_per_section]
            
            if top_chunks:
                reranked_results.append({
                    'section': section,
                    'pdf_name': pdf_name,
                    'chunks': top_chunks,
                    'avg_score': sum(c['relevance_score'] for c in top_chunks) / len(top_chunks),
                })
        
        # optionally sort sections by average relevance
        reranked_results.sort(key=lambda x: x['avg_score'], reverse=True)
        
        return reranked_results

    def rerank_hybrid(
        self,
        question: str,
        retrieved_results: List[Dict[str, Any]],
        total_chunks: int = 10,
        min_chunks_per_section: int = 1,
        max_chunks_per_section: int = 5,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        hybrid reranking: select best chunks globally while ensuring diversity
        guarantees at least min_chunks_per_section from each section
        
        :param question: user query
        :param retrieved_results: results from retriever
        :param total_chunks: total number of chunks to return
        :param min_chunks_per_section: minimum chunks to include per section
        :param max_chunks_per_section: maximum chunks from any single section
        :param min_score: optional minimum relevance score threshold
        :return: list of sections with reranked chunks
        """
        if not retrieved_results:
            return []
        
        # score all chunks
        all_scored_chunks = []
        # track chunks by section
        section_map = {}
        
        for result in retrieved_results:
            section = result.get('section', 'Unknown')
            pdf_name = result.get('pdf_name', 'Unknown')
            chunks = result.get('chunks', [])
            
            if not chunks:
                continue
            
            section_key = (section, pdf_name)
            section_map[section_key] = []
            
            # create query-document pairs
            pairs = [[question, chunk['text']] for chunk in chunks]
            
            # score all chunks
            scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
            
            for chunk, score in zip(chunks, scores):
                scored_chunk = {
                    'text': chunk['text'],
                    'chunk_id': chunk['chunk_id'],
                    'is_match': chunk.get('is_match', False),
                    'relevance_score': float(score),
                    'section': section,
                    'pdf_name': pdf_name,
                }
                all_scored_chunks.append(scored_chunk)
                section_map[section_key].append(scored_chunk)
        
        # sort all chunks by score
        all_scored_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # filter by minimum score
        if min_score is not None:
            all_scored_chunks = [c for c in all_scored_chunks if c['relevance_score'] >= min_score]
        
        print(f"Reranking {len(all_scored_chunks)} total chunks across {len(section_map)} sections...")
        
        # select chunks with diversity constraints
        selected_chunks = []
        section_counts = {k: 0 for k in section_map.keys()}
        
        # ensure minimum chunks per section
        for section_key in section_map.keys():
            section_chunks = section_map[section_key]
            section_chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            for chunk in section_chunks[:min_chunks_per_section]:
                if len(selected_chunks) < total_chunks:
                    selected_chunks.append(chunk)
                    section_counts[section_key] += 1
        
        # fill remaining slots with best chunks
        for chunk in all_scored_chunks:
            if len(selected_chunks) >= total_chunks:
                break
            
            section_key = (chunk['section'], chunk['pdf_name'])
            
            # skip if already selected
            if chunk in selected_chunks:
                continue
            
            # skip if section hit max
            if section_counts[section_key] >= max_chunks_per_section:
                continue
            
            selected_chunks.append(chunk)
            section_counts[section_key] += 1
        
        # organize by section
        result_sections = {}
        for chunk in selected_chunks:
            section_key = (chunk['section'], chunk['pdf_name'])
            if section_key not in result_sections:
                result_sections[section_key] = {
                    'section': chunk['section'],
                    'pdf_name': chunk['pdf_name'],
                    'chunks': [],
                }
            result_sections[section_key]['chunks'].append(chunk)
        
        # sort chunks within each section by chunk_id for readability
        final_results = []
        for section_data in result_sections.values():
            section_data['chunks'].sort(key=lambda x: x['chunk_id'])
            final_results.append(section_data)
        
        # sort sections by best chunk score
        final_results.sort(
            key=lambda x: max(c['relevance_score'] for c in x['chunks']), 
            reverse=True
        )
        
        print(f"Selected {len(selected_chunks)} chunks across {len(final_results)} sections")
        
        return final_results
