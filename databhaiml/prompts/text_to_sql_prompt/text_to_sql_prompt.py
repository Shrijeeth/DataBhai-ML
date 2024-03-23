"""
text_to_sql_prompt
~~~~~~~~~~~~~~~~~~

Module to prepare prompts for text to SQL conversion tasks.
"""

from ..prompt import Prompt

# pylint: disable=too-few-public-methods
class TextToSqlPrompt:
    """
    Class to prepare prompts for text to SQL conversion tasks.
    """

    def __init__(self, prompt_file_path):
        self.file_path = prompt_file_path
        self.prompt = Prompt(self.file_path)

    def get_prompt(self, question: str, schema: str, instructions: str, requirements: str):
        """
        Method to get the prompt for text to SQL conversion tasks.
        """

        prompt_kwargs = {
            "question": question,
            "schema": schema,
            "instructions": instructions,
            "requirements": requirements
        }
        return self.prompt.format(**prompt_kwargs)
