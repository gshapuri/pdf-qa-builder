from __future__ import annotations

from typing import List, Dict, Any

from pdf_pipeline.embedder import Embedder
from pdf_pipeline.vector_store import WeaviateVectorStore


class Retriever:
    """
    retriever for fetching semantically similar chunks from weaviate
    
    this class embeds a user query using the same embedding model used
    during indexing, performs a vector search against the weaviate database,
    and returns the top matching sections or chunks with metadata
    """

    def __init__(self, top_k: int = 3, embedder_model: str = "balanced"):
        """
        initialize retriever with embedding model and weaviate client
        
        :param top_k: number of results to return per query
        :param embedder_model: embedding model name (must match indexing model)
        """
        self.top_k = top_k
        self.embedder = Embedder(model_name=embedder_model)
        self.store = WeaviateVectorStore()

    def retrieve(self, question: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        retrieve top matching sections or chunks for a given question
        
        :param question: natural language query string
        :param top_k: override default number of results (optional)
        :return: list of search results with metadata
        """
        if not question or len(question.strip()) == 0:
            return []
        
        # use provided top_k or fall back to instance default
        k = top_k if top_k is not None else self.top_k
        
        print(f"Embedding question: '{question}'")
        query_vector = self.embedder.embed_texts([question])[0]
        
        print(f"Querying Weaviate for top {k} matches...")
        results = self.store.search_hybrid(query_vector, top_k=k)
        
        if not results:
            print("No results found")
            return []
        
        print(f"Retrieved {len(results)} results")
        return results

    def close(self):
        """
        close weaviate connection
        """
        self.store.close()
