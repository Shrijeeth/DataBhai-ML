"""
weaviate_vector_store
~~~~~~~~~~~~~~~~~~~~~

Module that provides functionality to interact with the Weaviate vector store.
"""

from abc import ABC
from typing import Optional, List, Dict

import weaviate
from langchain.vectorstores.weaviate import Weaviate
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)
from langchain_core.documents import Document

from .base_vector_store import BaseVectorStore


class WeaviateVectorStore(BaseVectorStore, ABC):
    """
    A class representing a Weaviate vector store.
    """

    def __init__(self, weaviate_url: str, index_name: str, api_key: Optional[str] = None):
        if api_key is None:
            self.client = weaviate.Client(weaviate_url)
        else:
            self.client = weaviate.Client(
                weaviate_url,
                auth_client_secret=weaviate.Auth.AuthApiKey(api_key=api_key)
            )
        self.index_name = index_name
        class_obj = {
            "classes": [
                {
                    "class": self.index_name,
                    "vectorizer": "text2vec-transformers",
                    "moduleConfig": {
                        "text2vec-transformers": {
                            "vectorizeClassName": False,
                            "inferenceUrl": "http://t2v-transformers:8080",
                        }
                    },
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "moduleConfig": {
                                "text2vec-transformers": {
                                    "skip": False,
                                    "vectorizePropertyName": False,
                                }
                            }
                        }
                    ]
                }
            ]
        }
        self.create_vector_store_schema(class_obj)
        self.vector_store = Weaviate(
            self.client,
            index_name=index_name,
            by_text=False,
            text_key="content"
        )

    def add_documents(self, documents):
        return self.vector_store.add_documents(documents)

    def add_texts(self, texts: List[str], metadata: Optional[List[Dict]] = None):
        formatted_documents = [
            Document(
                page_content=text,
                metadata=metadata[i] if metadata else None
            ) for i, text in enumerate(texts)
        ]
        return self.add_documents(formatted_documents)

    def search(self, query: str, k: Optional[int] = 5):
        return self.vector_store.similarity_search(query, k)

    def hybrid_search(self, query: str, where_filter: Dict, k: Optional[int] = 5):
        retriever = WeaviateHybridSearchRetriever(
            client=self.client,
            index_name=self.index_name[0].capitalize() + self.index_name[1:],
            text_key="content",
            create_schema_if_missing=False,
            k=k,
        )
        return retriever.get_relevant_documents(
            query,
            where_filter=where_filter,
        )

    def create_vector_store_schema(self, schema: Dict):
        """
        Create the vector store schema in Weaviate.

        Args:
            schema (dict): The schema to create.
        """

        if not self.client.schema.exists(self.index_name):
            self.client.schema.create(schema)

    def __del__(self):
        del self.client
