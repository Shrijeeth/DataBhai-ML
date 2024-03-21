from ..vector_stores.base_vector_store import BaseVectorStore
from enum import Enum

class TextToSqlTypes(Enum):
    TABLE = "TABLE"
    INSTRUCTION = "INSTRUCTION"
    REQUIREMENT = "REQUIREMENT"

class TextToSqlRag:
    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store

    def add_tables(self, table_name: str, table_schema: str, database_name: str, application_name: str):
        metadata = [
            {
                "table_name": table_name,
                "database_name": database_name,
                "application_name": application_name,
                "type": TextToSqlTypes.TABLE,
            },
        ]
        return self.vector_store.add_texts(table_schema, metadata=metadata)
    
    def add_instructions(self, instructions: str, database_name: str, application_name: str):
        metadata = [
            {
                "database_name": database_name,
                "application_name": application_name,
                "type": TextToSqlTypes.INSTRUCTION,
            },
        ]
        return self.vector_store.add_texts(instructions, metadata=metadata)
    
    def add_requirements(self, requirements: str, database_name: str, application_name: str):
        metadata = [
            {
                "database_name": database_name,
                "application_name": application_name,
                "type": TextToSqlTypes.REQUIREMENT,
            },
        ]
        return self.vector_store.add_texts(requirements, metadata=metadata)

    def get_similar_texts(self, query: str, top_k_tables: int = 3, top_k_instructions: int = 5, top_k_requirements: int = 2):
        tables = self.vector_store.hybrid_search(query, whereFilter={
            "type": TextToSqlTypes.TABLE,
        }, k=top_k_tables)
        table_prompt = ''
        for table in tables:
            table_prompt += table.page_content + "\n\n"

        instructions = self.vector_store.hybrid_search(query, whereFilter={
            "type": TextToSqlTypes.INSTRUCTION,
        }, k=top_k_instructions)
        instruction_prompt = ''
        for instruction in instructions:
            instruction_prompt += instruction.page_content + "\n\n"

        requirements = self.vector_store.hybrid_search(query, whereFilter={
            "type": TextToSqlTypes.REQUIREMENT,
        }, k=top_k_requirements)
        requirement_prompt = ''
        for requirement in requirements:
            requirement_prompt += requirement.page_content + "\n\n"

        return table_prompt, instruction_prompt, requirement_prompt
