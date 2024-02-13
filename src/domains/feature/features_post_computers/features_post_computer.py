from abc import ABC, abstractmethod

from domains.feature.feature_entry import FeatureEntry


class FeaturesPostComputer(ABC):
    @abstractmethod
    def compute(self, features: list[dict[str, FeatureEntry]]) -> list[dict[str, FeatureEntry]]:
        pass
