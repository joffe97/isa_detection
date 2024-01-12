from abc import ABC, abstractmethod


class FeatureComputer(ABC):
    @abstractmethod
    def compute(binary_file: str) -> dict[str, float]:
        pass
