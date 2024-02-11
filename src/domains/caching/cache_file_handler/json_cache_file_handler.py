from io import BufferedReader, BufferedWriter
import json
from .cache_file_handler import CacheFileHandler


class JsonCacheFileHandler(CacheFileHandler):
    def __init__(self) -> None:
        super().__init__("r", "w")

    @staticmethod
    def dump(data: object, writer: BufferedWriter) -> None:
        json.dump(data, writer)

    @staticmethod
    def load(reader: BufferedReader) -> object:
        return json.load(reader)
