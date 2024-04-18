from abc import ABC, abstractmethod
from pathlib import Path
from config import Config
from domains.dataset.binary_file_dataset import BinaryFileDataset


class CustomDataset(BinaryFileDataset, ABC):
    @classmethod
    def get_custom_file_path(cls) -> Path:
        return Config.CUSTOM_DATASET_PATH.joinpath(cls.class_name())

    @classmethod
    def open_custom_file(cls, mode: str):
        custom_dataset_dir_path = Config.CUSTOM_DATASET_PATH
        custom_dataset_dir_path.mkdir(parents=True, exist_ok=True)
        return open(cls.get_custom_file_path(), mode)

    @abstractmethod
    def create_data(self) -> bytes:
        pass

    def create_data_if_not_exists(self) -> None:
        if self.get_custom_file_path().exists():
            return

        data_bytes = self.create_data()
        with self.open_custom_file("wb") as f:
            f.write(data_bytes)

    def create_architecture_paths_mapping(self) -> dict[str, list[str]]:
        self.create_data_if_not_exists()
        return {self.class_name(): [str(self.get_custom_file_path())]}
