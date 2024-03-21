import abc
from typing import List, Optional, Dict


class BaseVectorStore(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "add_documents") and
            callable(subclass.add_documents) and
            hasattr(subclass, "add_texts") and
            callable(subclass.add_texts) and
            hasattr(subclass, "hybrid_search") and
            callable(subclass.hybrid_search) and
            hasattr(subclass, "search") and
            callable(subclass.search) or
            NotImplemented
        )

    @abc.abstractmethod
    def add_documents(self, documents):
        """
        Add documents to the vector store
        """
        raise NotImplementedError

    @abc.abstractmethod
    def add_texts(self, texts: str, metadata: Optional[List[Dict]] = None):
        """
        Add texts to the vector store
        """
        raise NotImplementedError

    @abc.abstractmethod
    def search(self, query: str, k: Optional[int] = 5):
        """
        Search the vector store for similar data
        """
        raise NotImplementedError

    @abc.abstractmethod
    def hybrid_search(self, query: str, whereFilter: Dict, k: Optional[int] = 5):
        """
        Search the vector store for similar data with metadata filters
        """
        raise NotImplementedError
