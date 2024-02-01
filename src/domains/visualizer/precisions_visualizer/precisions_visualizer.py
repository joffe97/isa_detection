from pandas import DataFrame
from abc import ABC, abstractmethod


class PrecisionsVisualizer(ABC):
    @abstractmethod
    def visualize(precisions: DataFrame):
        pass
