from abc import ABC
from typing import Literal, Optional, List, Dict
from langchain.vectorstores.weaviate import Weaviate
from langchain_community.retrievers import (
    WeaviateHybridSearchRetriever,
)
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings

from .base_vector_store import BaseVectorStore

import weaviate


class WeaviateVectorStore(BaseVectorStore, ABC):
    def __init__(self, weaviate_url: str, index_name: str, api_key: Optional[str] = None):
        if api_key is None:
            self.client = weaviate.Client(weaviate_url)
        else:
            self.client = weaviate.Client(weaviate_url, auth_client_secret=weaviate.Auth.AuthApiKey(api_key=api_key))
        self.num_results = num_results
        self.index_name = index_name
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-m3",
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vector_store = Weaviate(
            self.client,
            index_name=index_name,
            embedding=self.embeddings,
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
            index_name=self.index_name,
            embeddings=self.embeddings,
            text_key="content",
            create_schema_if_missing=True,
            k=k,
        )
        return retriever.get_relevant_documents(
            query,
            where_filter=where_filter,
        )

    def create_vector_store_schema(self, schema: Dict):
        self.client.schema.create(schema)

    def __del__(self):
        del self.client
        del self.embeddings
