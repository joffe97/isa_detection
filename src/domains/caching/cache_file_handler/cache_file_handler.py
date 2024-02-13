from abc import ABC, abstractmethod
from typing import IO


class CacheFileHandler(ABC):
    def __init__(self, read_mode: str, write_mode: str) -> None:
        super().__init__()
        self.read_mode = read_mode
        self.write_mode = write_mode

    @abstractmethod
    def dump(self, data: object, writer: IO) -> None:
        pass

    @abstractmethod
    def load(self, reader: IO) -> object:
        pass
