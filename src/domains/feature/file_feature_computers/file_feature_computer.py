from abc import ABC, abstractmethod


class FileFeatureComputer(ABC):
    @staticmethod
    @abstractmethod
    def compute(binary_file: str) -> dict[str, float]:
        pass
