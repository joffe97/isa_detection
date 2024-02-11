from io import BufferedReader, BufferedWriter
import pickle
from .cache_file_handler import CacheFileHandler


class PickleCacheFileHandler(CacheFileHandler):
    def __init__(self) -> None:
        super().__init__("rb", "wb")

    @staticmethod
    def dump(data: object, writer: BufferedWriter) -> None:
        pickle.dump(data, writer)

    @staticmethod
    def load(reader: BufferedReader) -> object:
        return pickle.load(reader)
