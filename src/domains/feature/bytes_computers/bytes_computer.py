from abc import ABC, abstractmethod
from typing import Literal, Optional, Union


class BytesComputer(ABC):
    @abstractmethod
    def compute(self, data: bytes) -> Union[list[float], list[int]]:
        pass

    @abstractmethod
    def get_group_name(
        self, data_identifiers: Optional[list[str]] = None, override_if_empty: str = ""
    ) -> str:
        pass

    @abstractmethod
    def labels(self) -> tuple[str, str]:
        pass

    def x_labels(self) -> Optional[list[int]]:
        return None

    def identifier(self) -> str:
        return self.__class__.mro()[0].__name__

    def y_scale(self) -> Literal["linear", "log", "symlog", "logit"]:
        return "linear"

    @classmethod
    def class_name(cls) -> str:
        return cls.__name__
