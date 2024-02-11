from abc import ABC, abstractmethod


class FeaturesPostComputer(ABC):
    @abstractmethod
    def compute(self, features: list[dict[str, object]]) -> list[dict[str, object]]:
        pass
