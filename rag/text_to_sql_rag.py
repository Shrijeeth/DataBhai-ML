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