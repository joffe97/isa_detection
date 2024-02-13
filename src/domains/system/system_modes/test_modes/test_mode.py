from abc import ABC, abstractmethod

from domains.model.info.isa_model_result_collection import ISAModelResultCollection
from domains.model.isa_model_collection import ISAModelCollection
from domains.model.isa_model_configuration import ISAModelConfiguration


class TestMode(ABC):
    @staticmethod
    @abstractmethod
    def run(
        isa_model_configuration: ISAModelConfiguration,
        isa_model_collection: ISAModelCollection,
    ) -> ISAModelResultCollection:
        pass
