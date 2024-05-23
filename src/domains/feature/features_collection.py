from typing import Union
from domains.feature.feature_entry import FeatureEntry


class FeaturesCollection(list[dict[str, FeatureEntry]]):
    def __init__(self):
        self.label_values_mappings: list[dict[str, Union[float, int]]] = []
        self.label_numerical_identifiers_mappings: dict[str, int] = dict()

    def append_numerical_features(
        self,
        label_values_mapping: dict[str, Union[float, int]],
        label_numerical_identifiers_mapping: dict[str, int],
    ):
        self.label_values_mappings.append(label_values_mapping)
        self.label_numerical_identifiers_mappings.update(label_numerical_identifiers_mapping)
