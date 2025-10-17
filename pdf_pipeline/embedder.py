from __future__ import annotations

from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

from pdf_pipeline.text_splitter import Chunk


# predefined model presets for different use cases
DEFAULT_MODELS = {
    "fast": "sentence-transformers/all-MiniLM-L6-v2",
    "balanced": "sentence-transformers/all-mpnet-base-v2",
    "scientific": "allenai/specter2_base",
}


class Embedder:
    """
    create embeddings for text chunks using a sentence-transformers model
    
    loads a hugging face model and provides methods to embed text chunks
    or arbitrary lists of strings. supports model presets for easy switching
    """

    def __init__(
        self,
        model_name: str = "scientific",
        batch_size: int = 16,
        device: str = "cpu",
    ):
        """
        initialize embedder with a sentence-transformer model
        
        :param model_name: model preset ('fast', 'balanced', 'scientific') or full model path
        :param batch_size: number of texts to embed per batch
        :param device: device to use ('cpu' or 'cuda' if available)
        """
        # resolve preset names to actual model paths
        if model_name in DEFAULT_MODELS:
            resolved_model = DEFAULT_MODELS[model_name]
        else:
            resolved_model = model_name
        
        self.model_name = resolved_model
        self.batch_size = batch_size
        self.device = device
        
        # load model with trust_remote_code for some models
        try:
            self.model = SentenceTransformer(resolved_model, device=device)
        except Exception as e:
            # fallback to balanced model if scientific model fails
            print(f"Failed to load {resolved_model}: {e}")
            print("Falling back to balanced model...")
            self.model_name = DEFAULT_MODELS["balanced"]
            self.model = SentenceTransformer(self.model_name, device=device)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        create embeddings for a list of text strings
        
        :param texts: list of text strings
        :return: list of embedding vectors
        """
        if not texts:
            return []
        
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        
        return vectors.tolist()

    def embed_chunks(self, chunks: List[Chunk]) -> List[Dict[str, Any]]:
        """
        embed a list of chunk objects and attach vectors to them
        
        :param chunks: list of chunk objects
        :return: list of dicts containing section, chunk_id, text, and vector
        """
        if not chunks:
            return []
        
        texts = [c.text for c in chunks]
        vectors = self.embed_texts(texts)
        
        enriched: List[Dict[str, Any]] = []
        for chunk, vector in zip(chunks, vectors):
            enriched.append({
                "section": chunk.section,
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "vector": vector,
            })
        
        return enriched
