from abc import ABC, abstractmethod

from domains.feature.feature_entry import FeatureEntry


class FileFeatureComputer(ABC):
    @staticmethod
    @abstractmethod
    def compute(binary_file: str) -> dict[str, FeatureEntry]:
        pass
