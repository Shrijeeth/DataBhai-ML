from ..prompt import Prompt

class TextToSqlPrompt:
    def __init__(self, version: int = 1):
        self.file_path = self.get_file_path(version)
        self.prompt = Prompt(self.file_path)

    def get_file_path(self, version: int):
        return r"./templates/v1.txt"
    
    def get_prompt(self, question: str, schema: str, instructions: str, requirements: str):
        return self.prompt.format(question, schema, instructions, requirements)