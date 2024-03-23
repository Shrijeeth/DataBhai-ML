"""
base_prompt_template
~~~~~~~~~~~~~~~~~~~~

Module that provides a base abstract class to interact with the Prompt Templates.
"""

import abc

class BasePromptTemplate(metaclass=abc.ABCMeta):
    """
    Base abstract class to interact with the Prompt Templates.
    """

    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "format")
            and callable(subclass.format)
        )

    @abc.abstractmethod
    def format(self, **kwargs) -> str:
        """
        Method to format prompts based on prompt variables provided
        """
        raise NotImplementedError
