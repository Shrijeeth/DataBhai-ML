import abc

class BasePromptTemplate(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, subclass):
        return (
            hasattr(subclass, "format")
            and callable(subclass.format)
        )

    @abc.abstractmethod
    def format(self, **kwargs) -> str:
        pass