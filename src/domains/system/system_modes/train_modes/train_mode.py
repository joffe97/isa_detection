from abc import ABC, abstractmethod

from domains.model.isa_model_collection import ISAModelCollection
from domains.model.isa_model_configuration import ISAModelConfiguration


class TrainMode(ABC):
    @staticmethod
    @abstractmethod
    def run(isa_model_configuration: ISAModelConfiguration) -> ISAModelCollection:
        pass
