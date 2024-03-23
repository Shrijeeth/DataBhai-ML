"""
text_to_sql_rag
~~~~~~~~~~~~~~~

Module that provides RAG functionalities for Text To SQL Tasks
"""

from enum import Enum
from databhaiml.vector_stores.base_vector_store import BaseVectorStore

class TextToSqlTypes(Enum):
    """
    Enum class to represent the different types of text-to-SQL Data Types.
    """
    TABLE = "TABLE"
    INSTRUCTION = "INSTRUCTION"
    REQUIREMENT = "REQUIREMENT"

class TextToSqlRag:
    """
    Class to handle text-to-SQL tasks using RAG.
    """

    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store

    def add_tables(
        self,
        table_name: str,
        table_schema: str,
        database_name: str,
        application_name: str
    ):
        """
        Add tables to the vector store.
        """

        metadata = [
            {
                "table_name": table_name,
                "database_name": database_name,
                "application_name": application_name,
                "type": str(TextToSqlTypes.TABLE),
            },
        ]
        return self.vector_store.add_texts([table_schema], metadata=metadata)

    def add_instructions(self, instructions: str, database_name: str, application_name: str):
        """
        Add DB related general instructions to the vector store.
        """

        metadata = [
            {
                "database_name": database_name,
                "application_name": application_name,
                "type": str(TextToSqlTypes.INSTRUCTION),
            },
        ]
        return self.vector_store.add_texts([instructions], metadata=metadata)

    def add_requirements(self, requirements: str, database_name: str, application_name: str):
        """
        Add previous requirements and queries solved to the vector store.
        """
        metadata = [
            {
                "database_name": database_name,
                "application_name": application_name,
                "type": str(TextToSqlTypes.REQUIREMENT),
            },
        ]
        return self.vector_store.add_texts([requirements], metadata=metadata)

    def get_similar_texts(
        self,
        query: str,
        top_k_tables: int = 3,
        top_k_instructions: int = 5,
        top_k_requirements: int = 2
    ):
        """
        Get similar texts for the query.

        Args:
            query (str): The query to search for similar texts.
            top_k_tables (int, optional): Number of tables to return. Defaults to 3.
            top_k_instructions (int, optional): Number of instructions to return. Defaults to 5.
            top_k_requirements (int, optional): Number of requirements to return. Defaults to 2.
        
        Returns:
            list: List of similar texts.
        """

        tables = self.vector_store.hybrid_search(query, where_filter=dict({
            "path": ["type"],
            "operator": "Equal",
            "valueString": str(TextToSqlTypes.TABLE),
        }), k=top_k_tables)
        table_prompt = ''
        for table in tables:
            table_prompt += table.page_content + "\n\n"

        instructions = self.vector_store.hybrid_search(query, where_filter=dict({
            "path": ["type"],
            "operator": "Equal",
            "valueString": str(TextToSqlTypes.INSTRUCTION),
        }), k=top_k_instructions)
        instruction_prompt = ''
        for instruction in instructions:
            instruction_prompt += instruction.page_content + "\n\n"

        requirements = self.vector_store.hybrid_search(query, where_filter=dict({
            "path": ["type"],
            "operator": "Equal",
            "valueString": str(TextToSqlTypes.REQUIREMENT),
        }), k=top_k_requirements)
        requirement_prompt = ''
        for requirement in requirements:
            requirement_prompt += requirement.page_content + "\n\n"

        return table_prompt, instruction_prompt, requirement_prompt
