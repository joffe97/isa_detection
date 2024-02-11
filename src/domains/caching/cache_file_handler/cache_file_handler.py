from abc import ABC, abstractmethod
from io import BufferedReader, BufferedWriter


class CacheFileHandler(ABC):
    def __init__(self, read_mode: str, write_mode: str) -> None:
        super().__init__()
        self.read_mode = read_mode
        self.write_mode = write_mode

    @staticmethod
    @abstractmethod
    def dump(data: object, writer: BufferedWriter) -> None:
        pass

    @staticmethod
    @abstractmethod
    def load(reader: BufferedReader) -> object:
        pass
