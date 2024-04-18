from abc import ABC, abstractmethod
from pathlib import Path
import itertools

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
    ) -> Path:
        time_directory_name = Config.get_readable_start_datetime() if use_time_directory else None

        research_class_path_directories_optional = [
            cls._class_name(),
            group_name,
            time_directory_name,
            filename,
        ]
        research_class_path_directories = filter(None, research_class_path_directories_optional)

        return Config.RESEARCH_PATH.joinpath(*research_class_path_directories).with_suffix(suffix)

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
