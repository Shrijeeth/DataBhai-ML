from llama_cpp import Llama
from typing import Union, List

import abc

import transformers


class BaseTextModel(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
                hasattr(subclass, 'load_model') and
                callable(subclass.load_model) and
                hasattr(subclass, 'load_tokenizer') and
                callable(subclass.load_tokenizer) and
                hasattr(subclass, 'generate') and
                callable(subclass.generate) or
                NotImplemented
        )

    @abc.abstractmethod
    def load_model(self, model_path: str, **kwargs) -> Union[transformers.PreTrainedModel, Llama]:
        """Method to Load corresponding ML/DL Models"""
        raise NotImplementedError

    @abc.abstractmethod
    def load_tokenizer(self, tokenizer_path: str, **kwargs) -> Union[transformers.PreTrainedTokenizerBase, None]:
        """Method to Load tokenizer for the model to tokenize characters"""
        raise NotImplementedError

    @abc.abstractmethod
    def generate(self, inputs: str, max_new_tokens: int, num_beams: int, **kwargs) -> str:
        """Method to generate outputs for the text model based on input texts"""
        raise NotImplementedError