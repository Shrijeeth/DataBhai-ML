from ..prompt import Prompt

class TextToSqlPrompt:
    def __init__(self, prompt_file_path):
        self.file_path = prompt_file_path
        self.prompt = Prompt(self.file_path)
    
    def get_prompt(self, question: str, schema: str, instructions: str, requirements: str):
        prompt_kwargs = {
            "question": question,
            "schema": schema,
            "instructions": instructions,
            "requirements": requirements
        }
        return self.prompt.format(**prompt_kwargs)
