from pathlib import Path
from typing import Any, Optional


class FileData:
    def __init__(self, path: Optional[Path], data: Any) -> None:
        self.path = path
        self.data = data

    @classmethod
    def with_none_path(cls, data: Any) -> "FileData":
        return cls(None, data)

    @property
    def filename(self) -> str:
        return self.path.name if self.path is not None else "None"

    def file_data_mapping(self) -> tuple[str, list[float]]:
        return (self.filename, self.data)
