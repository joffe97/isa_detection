from domains.feature.feature_entry import FeatureEntry
from domains.feature.features_post_computers.features_post_computer import (
    FeaturesPostComputer,
)


class NoPostComputing(FeaturesPostComputer):
    def compute(self, features: list[dict[str, FeatureEntry]]) -> list[dict[str, float]]:
        return [
            dict((key, feature_entry.value) for key, feature_entry in features_dict.items())
            for features_dict in features
        ]

    def identifier(self) -> str:
        return ""
