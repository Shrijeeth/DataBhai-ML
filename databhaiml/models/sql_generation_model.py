"""
sql_generation_model
~~~~~~~~~~~~~~~~~~~~

Module that provides functionalities to load and process data 
using LLM for SQL Generation Tasks
"""

from typing import Union
from abc import ABC

import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama

from .base_model import BaseTextModel


class SqlGenerationModel(BaseTextModel, ABC):
    """
    Class to load and process data 
    using LLM for SQL Generation Tasks which uses Base Text Model
    """

    def __init__(self, model_path: str, tokenizer_path: str, is_optimized: bool = False, **kwargs):
        super().__init__()
        self.is_optimized = is_optimized
        self.model = self.load_model(model_path, **kwargs)
        self.tokenizer = self.load_tokenizer(tokenizer_path, **kwargs)

    def load_model(self, model_path: str, **kwargs) -> Union[transformers.PreTrainedModel, Llama]:
        if self.is_optimized:
            self.model = Llama(model_path=model_path, n_ctx=8192, n_threads=7, n_batch=512)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            self.model.config.use_cache = False
        return self.model

    def load_tokenizer(
        self,
        tokenizer_path: str,
        **kwargs
    ) -> Union[transformers.PreTrainedTokenizerBase, None]:
        if self.is_optimized:
            return None
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer = tokenizer
        return tokenizer

    def generate(self, inputs: str, max_new_tokens: int = 400, num_beams: int = 1, **kwargs) -> str:
        if self.is_optimized:
            outputs = self.model(inputs, echo=False, stream=False, max_tokens=max_new_tokens)
            return outputs['choices'][0]['text']

        if (self.tokenizer is None) or (self.model is None):
            raise ModuleNotFoundError("Model and Tokenizer must be loaded before text generation")
        tokens = self.tokenizer(inputs, return_tensors="pt")
        generated_ids = self.model.generate(
            **tokens,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            do_sample=False,
        )
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        return outputs[0]
