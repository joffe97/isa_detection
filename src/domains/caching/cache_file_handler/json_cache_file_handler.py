from typing import IO
import jsonpickle
from .cache_file_handler import CacheFileHandler


class JsonCacheFileHandler(CacheFileHandler):
    def __init__(self, indent=None) -> None:
        super().__init__("r", "w")
        self.indent = indent

    def dump(self, data: object, writer: IO) -> None:
        json_str: bytes = jsonpickle.encode(data, indent=self.indent)  # type: ignore
        writer.write(json_str)

    def load(self, reader: IO) -> object:
        json_str = reader.read()
        return jsonpickle.decode(json_str)
