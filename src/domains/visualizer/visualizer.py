from domains.model.isa_model import ISAModel
from abc import ABC, abstractmethod


class ISAModelVisaulizer(ABC):
    @abstractmethod
    def visualize(isa_model: ISAModel):
        pass
