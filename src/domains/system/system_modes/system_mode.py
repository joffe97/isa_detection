from abc import ABC, abstractmethod
from sklearn.base import ClassifierMixin

from domains.feature.feature_computer_collection import FeatureComputerCollection


class SystemMode(ABC):
    @abstractmethod
    def run(self, feature_computer_collection: FeatureComputerCollection, classifier: ClassifierMixin, files_per_architecture: int):
        pass
