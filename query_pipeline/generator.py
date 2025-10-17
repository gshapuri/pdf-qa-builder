from __future__ import annotations

import os
from typing import List, Dict, Any

from openai import OpenAI


class Generator:
    """
    generator for creating final answers from retrieved chunks
    
    combines the user question and retrieved context into a single prompt
    and uses an llm to produce a concise, factual answer
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        """
        initialize generator with openai client
        
        :param model_name: openai model to use
        :param temperature: sampling temperature (0.0 = deterministic)
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model_name = model_name
        self.temperature = temperature

    def generate(
        self, 
        question: str, 
        retrieved_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        generate a final answer using context from retrieved results
        
        :param question: user query
        :param retrieved_results: list of search results from retriever
        :return: dict with answer and metadata
        """
        if not retrieved_results:
            return {
                "answer": "No relevant context found to answer the question.",
                "sources": [],
            }
        
        # extract and format context from retrieved results
        context_parts = []
        sources = []
        
        for i, result in enumerate(retrieved_results, 1):
            section = result.get('section', 'Unknown')
            pdf_name = result.get('pdf_name', 'Unknown')
            chunks = result.get('chunks', [])
            
            # combine all chunks from this section
            section_text = "\n".join(chunk['text'] for chunk in chunks)
            
            context_parts.append(
                f"[Source {i}: {pdf_name} - {section}]\n{section_text}"
            )
            
            sources.append({
                'pdf': pdf_name,
                'section': section,
                'num_chunks': len(chunks),
            })
        
        # join all context
        full_context = "\n\n".join(context_parts)
        
        # construct the prompt
        system_prompt = (
            "You are a helpful research assistant. Answer questions based strictly on the provided context. "
            "If the context doesn't contain enough information, say so. "
            "Cite sources by referring to [Source N] when making claims."
        )
        
        user_prompt = (
            f"Question: {question}\n\n"
            f"Context:\n{full_context}\n\n"
            f"Provide a clear, concise answer based on the context above."
        )
        
        # call openai api
        print(f"Generating answer with {self.model_name}...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
            )
            
            answer = response.choices[0].message.content
            
            return {
                "answer": answer,
                "sources": sources,
                "model": self.model_name,
            }
            
        except Exception as e:
            print(f"Error generating answer: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": sources,
            }

    def close(self):
        """
        close openai client connection

        :return:
        """
        if hasattr(self.client, 'close'):
            self.client.close()