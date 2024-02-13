from abc import ABC, abstractmethod
from pandas import DataFrame


class PrecisionsVisualizer(ABC):
    @staticmethod
    @abstractmethod
    def visualize(precisions: DataFrame):
        pass
