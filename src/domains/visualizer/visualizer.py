from abc import ABC, abstractmethod
from domains.model.isa_model import ISAModel


class ISAModelVisaulizer(ABC):
    @staticmethod
    @abstractmethod
    def visualize(isa_model: ISAModel):
        pass
