from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.init import AdditionalConfig, Timeout


class WeaviateVectorStore:
    """
    interface for interacting with a weaviate vector database

    stores section summaries AND chunk embeddings for hybrid search
    summaries are only used for matching, chunks are returned
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        skip_init_checks: bool = False,
    ):
        """
        initialize weaviate client connection

        :param url: weaviate endpoint url (default: http://localhost:8080)
        :param api_key: optional api key for cloud instance
        :param skip_init_checks: skip connection checks on init
        """
        self.url = url or os.getenv("WEAVIATE_URL", "http://localhost:8080")
        self.api_key = api_key or os.getenv("WEAVIATE_API_KEY")
        self.summary_collection = "Summary"
        self.chunk_collection = "Chunk"

        try:
            if self.api_key:
                self.client = weaviate.connect_to_weaviate_cloud(
                    cluster_url=self.url,
                    auth_credentials=weaviate.auth.AuthApiKey(self.api_key),
                    skip_init_checks=skip_init_checks,
                )
                print(f"Connected to Weaviate cloud: {self.url}")
            else:
                host = self.url.replace("http://", "").replace("https://", "").split(":")[0]
                
                self.client = weaviate.connect_to_local(
                    host=host,
                    port=8080,
                    grpc_port=50051,
                    skip_init_checks=skip_init_checks,
                    additional_config=AdditionalConfig(
                        timeout=Timeout(init=10, query=30, insert=30)
                    )
                )
                print(f"Connected to local Weaviate at {self.url}")
            
            if not skip_init_checks:
                self.client.is_ready()
                
        except Exception as e:
            print(f"Error: Could not connect to Weaviate at {self.url}")
            print(f"Make sure Weaviate is running:")
            print(f"  - docker-compose up -d")
            raise ConnectionError(f"Failed to connect to Weaviate: {e}")

    def ensure_schema(self):
        """
        create both summary and chunk collections if they don't exist

        :return:
        """
        try:
            # create summary collection
            if not self.client.collections.exists(self.summary_collection):
                self.client.collections.create(
                    name=self.summary_collection,
                    description="Summaries with embeddings (for matching only, not returned)",
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="section", data_type=DataType.TEXT),
                        Property(name="summary_key", data_type=DataType.TEXT),
                        Property(name="summary_text", data_type=DataType.TEXT),
                        Property(name="pdf_name", data_type=DataType.TEXT),
                        Property(name="summary_level", data_type=DataType.TEXT),
                    ],
                )
                print(f"Created collection '{self.summary_collection}'")
            
            # create chunk collection with vectors
            if not self.client.collections.exists(self.chunk_collection):
                self.client.collections.create(
                    name=self.chunk_collection,
                    description="Full text chunks with embeddings",
                    vectorizer_config=Configure.Vectorizer.none(),
                    properties=[
                        Property(name="section", data_type=DataType.TEXT),
                        Property(name="chunk_id", data_type=DataType.INT),
                        Property(name="text", data_type=DataType.TEXT),
                        Property(name="pdf_name", data_type=DataType.TEXT),
                    ],
                )
                print(f"Created collection '{self.chunk_collection}'")
                
        except Exception as e:
            print(f"Error ensuring schema: {e}")
            raise

    def insert_sections(
        self, 
        sections: Dict[str, str],
        summaries: Dict[str, str],
        summary_vectors: Dict[str, List[float]],
        chunks: List[Dict[str, Any]],
        chunk_vectors: List[List[float]],
        pdf_name: str,
        summary_level: str = "section",
    ):
        """
        insert summaries and chunks with their embeddings

        :param sections: dict of section names to full text
        :param summaries: dict of keys to summary text
        :param summary_vectors: dict of keys to embedding vectors
        :param chunks: list of chunk dicts with section, chunk_id, text
        :param chunk_vectors: list of embedding vectors for chunks
        :param pdf_name: name of the pdf file
        :param summary_level: 'section' or 'chunk'
        :return:
        """
        self.ensure_schema()

        try:
            # insert summaries
            summary_collection = self.client.collections.get(self.summary_collection)
            with summary_collection.batch.dynamic() as batch:
                for summary_key, summary_text in summaries.items():
                    # extract section name from key
                    if summary_level == "section":
                        section_name = summary_key
                    else:  # chunk level
                        # key format: "section_name_chunk_N"
                        section_name = "_".join(summary_key.split("_")[:-2])
                    
                    batch.add_object(
                        properties={
                            "section": section_name,
                            "summary_key": summary_key,
                            "summary_text": summary_text,
                            "pdf_name": pdf_name,
                            "summary_level": summary_level,
                        },
                        vector=summary_vectors[summary_key],
                    )
            
            print(f"Inserted {len(summaries)} summaries")
            
            # insert chunks with vectors
            chunk_collection = self.client.collections.get(self.chunk_collection)
            with chunk_collection.batch.dynamic() as batch:
                for chunk, vector in zip(chunks, chunk_vectors):
                    batch.add_object(
                        properties={
                            "section": chunk["section"],
                            "chunk_id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "pdf_name": pdf_name,
                        },
                        vector=vector,
                    )
            
            print(f"Inserted {len(chunks)} chunks with embeddings")
            
        except Exception as e:
            print(f"Error inserting data: {e}")
            raise

    def search_hybrid(
        self, 
        query_vector: List[float], 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        hybrid search: search both summaries and chunks, return full section chunks
        summaries are only used for matching, never returned to user

        :param query_vector: embedding vector for the query
        :param top_k: number of sections to return
        :return: list of results with section info and all chunks
        """
        try:
            matching_sections = set()
            matching_chunk_ids = {}  # track which chunks matched directly
            
            # search summaries
            summary_collection = self.client.collections.get(self.summary_collection)
            summary_response = summary_collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k * 2,
                return_properties=["section", "pdf_name", "summary_level"],
            )
            
            for obj in summary_response.objects:
                section_name = obj.properties.get("section", "")
                pdf_name = obj.properties.get("pdf_name", "")
                matching_sections.add((section_name, pdf_name))
            
            # search chunks
            chunk_collection = self.client.collections.get(self.chunk_collection)
            chunk_response = chunk_collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k * 3,
                return_properties=["section", "chunk_id", "pdf_name"],
            )
            
            for obj in chunk_response.objects:
                section_name = obj.properties.get("section", "")
                pdf_name = obj.properties.get("pdf_name", "")
                chunk_id = obj.properties.get("chunk_id", 0)
                
                matching_sections.add((section_name, pdf_name))
                
                # track which chunks matched directly
                key = (section_name, pdf_name)
                if key not in matching_chunk_ids:
                    matching_chunk_ids[key] = set()
                matching_chunk_ids[key].add(chunk_id)
            
            # build results: get all chunks for matching sections
            results = []
            
            for section_name, pdf_name in list(matching_sections)[:top_k]:
                # get all chunks for this section
                chunks_response = chunk_collection.query.fetch_objects(
                    filters=weaviate.classes.query.Filter.by_property("section").equal(section_name) &
                           weaviate.classes.query.Filter.by_property("pdf_name").equal(pdf_name),
                    limit=1000,
                )
                
                chunks = []
                for chunk_obj in chunks_response.objects:
                    chunk_id = chunk_obj.properties.get("chunk_id", 0)
                    chunk_data = {
                        "chunk_id": chunk_id,
                        "text": chunk_obj.properties.get("text", ""),
                    }
                    
                    # mark if this chunk was a direct match
                    key = (section_name, pdf_name)
                    chunk_data["is_match"] = key in matching_chunk_ids and chunk_id in matching_chunk_ids[key]
                    
                    chunks.append(chunk_data)
                
                # sort chunks by chunk_id
                chunks.sort(key=lambda x: x["chunk_id"])
                
                results.append({
                    "section": section_name,
                    "pdf_name": pdf_name,
                    "chunks": chunks,
                })
            
            return results
            
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def close(self):
        """
        close the weaviate client connection

        :return:
        """
        if hasattr(self, 'client'):
            self.client.close()
            print("Connection closed")
