import pickle
from typing import IO
from .cache_file_handler import CacheFileHandler


class PickleCacheFileHandler(CacheFileHandler):
    def __init__(self) -> None:
        super().__init__("rb", "wb")

    def dump(self, data: object, writer: IO) -> None:
        pickle.dump(data, writer)

    def load(self, reader: IO) -> object:
        return pickle.load(reader)
