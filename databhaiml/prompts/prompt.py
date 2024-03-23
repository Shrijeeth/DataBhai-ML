"""
prompt
~~~~~~

Module to format prompts based on prompt provided from text file
"""

from abc import ABC
from langchain import PromptTemplate

from .base_prompt_template import BasePromptTemplate

# pylint: disable=too-few-public-methods
class Prompt(BasePromptTemplate, ABC):
    """
    Class to format prompts based on prompt provided from text file
    """

    def __init__(self, prompt_file_path):
        super().__init__()
        self.prompt_template = ''
        with open(file=prompt_file_path, mode='r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        self.prompt = PromptTemplate.from_template(template=self.prompt_template)

    def format(self, **kwargs) -> str:
        return self.prompt.format(**kwargs)
