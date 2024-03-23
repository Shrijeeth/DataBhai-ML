from abc import ABC
from langchain import PromptTemplate

from .base_prompt_template import BasePromptTemplate


class Prompt(BasePromptTemplate, ABC):
    def __init__(self, prompt_file_path):
        super().__init__()
        self.prompt_template = ''
        with open(prompt_file_path, 'r') as f:
            self.prompt_template = f.read()
        self.prompt = PromptTemplate.from_template(template=self.prompt_template)

    def format(self, **kwargs) -> str:
        return self.prompt.format(**kwargs)
