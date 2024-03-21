from typing import List

import transformers
import torch

from abc import ABC
from .base_model import BaseTextModel
from ..utils import get_device_type
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_cpp import Llama


torch.set_default_device(get_device_type())


class SqlGenerationModel(BaseTextModel, ABC):
    def __init__(self, model_path: str, tokenizer_path: str, is_optimized: bool = False, **kwargs):
        super().__init__()
        self.is_optimized = is_optimized
        self.model = self.load_model(model_path, **kwargs)
        self.tokenizer = self.load_tokenizer(tokenizer_path, **kwargs)

    def load_model(self, model_path: str, **kwargs) -> transformers.PreTrainedModel | Llama:
        if self.is_optimized:
            self.model = Llama(model_path=model_path, n_ctx=2048, n_threads=5, n_batch=512)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        return self.model

    def load_tokenizer(self, tokenizer_path: str, **kwargs) -> transformers.PreTrainedTokenizerBase | None:
        if self.is_optimized:
            return None
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer = tokenizer
        return tokenizer

    def generate(self, inputs: str, max_new_tokens: int = 400, num_beams: int = 1, **kwargs) -> str:
        if self.is_optimized:
            outputs = self.model(inputs, echo=True, stream=False, max_tokens=8192)
            return outputs['choices'][0]['text']
        else:
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
