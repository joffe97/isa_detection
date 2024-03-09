from abc import ABC, abstractmethod


class BinaryFileDataset(ABC):
    @abstractmethod
    def create_architecture_paths_mapping(self) -> dict[str, list[str]]:
        pass
