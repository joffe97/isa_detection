from abc import ABC, abstractmethod
import json
from typing import Optional

import xxhash

from domains.feature.feature_entry import FeatureEntry


class FeaturesPostComputer(ABC):
    @abstractmethod
    def compute(self, features: list[dict[str, FeatureEntry]]) -> list[dict[str, float]]:
        pass

    def identifier(self):
        class_name = self.__class__.__name__
        attributes = self.__dict__

        def get_attribute_str_if_short() -> Optional[str]:
            no_hash_types = [str, int, float]
            if (
                len(attributes) == 1
                and any(
                    isinstance(first_attribute_value := list(attributes.values())[0], no_hash_type)
                    for no_hash_type in no_hash_types
                )
                and len(first_attribute_value_str := str(first_attribute_value)) <= 16
            ):
                return first_attribute_value_str
            return None

        attributes_str = (
            first_attribute_str
            if (first_attribute_str := get_attribute_str_if_short()) is not None
            else xxhash.xxh32(json.dumps(attributes)).hexdigest()
        )
        return f"{class_name}{attributes_str}"
