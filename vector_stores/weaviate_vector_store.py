from abc import ABC
from typing import Optional, List, Dict
from langchain.vectorstores.weaviate import Weaviate
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
        self.index_name = index_name
        self.text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
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
        formatted_docs = self.text_splitter.split_documents(documents)
        return self.vector_store.add_documents(formatted_docs)

    def add_texts(self, texts: str, metadata: Optional[List[Dict]] = None):
        formatted_texts = self.text_splitter.split_text(texts)
        return self.vector_store.add_texts(formatted_texts, metadatas=metadata)

    def search(self, query: str, k: Optional[int] = 5):
        docs = self.vector_store.similarity_search(query, k)
        return docs

    def create_vector_store_schema(self, schema: Dict):
        self.client.schema.create(schema)

    def __del__(self):
        del self.client
        del self.embeddings
