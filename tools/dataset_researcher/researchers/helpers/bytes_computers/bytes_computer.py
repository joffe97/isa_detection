from abc import ABC, abstractmethod
from typing import Optional


class BytesComputer(ABC):
    @abstractmethod
    def compute(self, data: bytes) -> list[float]:
        pass

    @abstractmethod
    def get_group_name(self, data_identifiers: Optional[list[str]] = None) -> str:
        pass

    def identifier(self) -> str:
        return self.__class__.mro()[0].__name__
