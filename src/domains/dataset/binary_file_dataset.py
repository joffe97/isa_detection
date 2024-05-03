from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Generator, Iterator

from helpers.get_object_variables import get_object_variables
from domains.dataset.classes.architecture_file_datas import ArchitectureFileDatas
from domains.dataset.classes.architecture_file_datas_mapping import ArchitectureFileDatasMapping

from .classes.file_data import FileData


class BinaryFileDataset(ABC):
    @classmethod
    def class_name(cls) -> str:
        return cls.__name__

    def identifier(self) -> str:
        class_attributes = get_object_variables(self)
        attribute_values = [str(value) for _, value in class_attributes]
        return "_".join([self.class_name(), *attribute_values])

    @abstractmethod
    def create_architecture_paths_mapping(self) -> dict[str, list[str]]:
        pass

    def iter_architectures_with_files_data(
        self, byte_read_count: int
    ) -> Generator[tuple[str, Generator[tuple[Path, bytes], None, None]], None, None]:
        def iter_paths(paths: list[Path]) -> Generator[tuple[Path, bytes], None, None]:
            for path in paths:
                with open(path, "rb") as f:
                    data = f.read(byte_read_count)
                yield (path, data)

        for architecture, path_strs in self.create_architecture_paths_mapping().items():
            paths = list(map(Path, path_strs))
            yield (architecture, iter_paths(paths))

    def create_architecture_func_data_mapping(
        self, byte_read_count: int, func: Callable[[bytes], Any]
    ) -> ArchitectureFileDatasMapping:
        architecture_mapping: dict[str, ArchitectureFileDatas] = dict()
        for architecture, files_data_iter in self.iter_architectures_with_files_data(byte_read_count):
            for path, file_data in files_data_iter:
                func_data = func(file_data)

                architecture_mapping.setdefault(architecture, ArchitectureFileDatas(architecture, []))
                architecture_mapping[architecture].append_file_data(FileData(path, func_data))
        return ArchitectureFileDatasMapping(architecture_mapping)
