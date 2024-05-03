from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from config import Config
from domains.dataset.binary_file_dataset import BinaryFileDataset


class Researcher(ABC):
    @classmethod
    def _class_name(cls) -> str:
        return cls.mro()[0].__name__

    @classmethod
    def _create_result_path(
        cls,
        group_name: str,
        filename: str,
        suffix: str,
        use_time_directory: bool = False,
        class_name_override: Optional[str] = None,
    ) -> Path:
        time_directory_name = Config.get_readable_start_datetime() if use_time_directory else None

        class_identifier = class_name_override or cls._class_name()
        if class_name_override is not None:
            class_identifier = class_name_override

        research_class_path_directories_optional = [
            class_identifier,
            group_name,
            time_directory_name,
            filename,
        ]
        research_class_path_directories = filter(None, research_class_path_directories_optional)

        path = Config.RESEARCH_PATH.joinpath(*research_class_path_directories)
        return path.with_suffix(path.suffix + suffix)

    @classmethod
    def _create_result_path_with_architecture_and_binary_file(
        cls,
        group_name: str,
        architecture: str,
        binary_file_name: str,
        suffix: str,
        use_time_directory: bool = False,
    ) -> Path:
        filename = "_".join([architecture, binary_file_name])
        return cls._create_result_path(group_name, filename, suffix, use_time_directory)

    @abstractmethod
    def research(self, dataset: BinaryFileDataset):
        pass
