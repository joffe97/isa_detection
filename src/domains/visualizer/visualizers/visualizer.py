from abc import ABC, abstractmethod
from domains.model.info.isa_model_info_collection import ISAModelInfoCollection


class Visualizer(ABC):
    @classmethod
    def class_name(cls) -> str:
        return cls.mro()[0].__name__

    @staticmethod
    @abstractmethod
    def get_str(isa_model_info_collection: ISAModelInfoCollection) -> str:
        pass

    @classmethod
    def visualize(cls, isa_model_info_collection: ISAModelInfoCollection):
        print(cls.get_str(isa_model_info_collection))
