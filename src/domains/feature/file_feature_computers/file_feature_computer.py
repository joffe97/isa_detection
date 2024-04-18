from abc import ABC, abstractmethod

from domains.feature.feature_entry import FeatureEntry
from helpers.get_object_variables import get_object_variables


class FileFeatureComputer(ABC):
    @abstractmethod
    def compute(self, binary_file: str) -> dict[str, FeatureEntry]:
        pass

    def identifier(self) -> str:
        class_name = self.__class__.__name__
        class_attributes = get_object_variables(self)
        attribute_values = [str(value) for _, value in class_attributes]
        return "_".join([class_name, *attribute_values])
