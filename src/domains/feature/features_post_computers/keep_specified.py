from domains.feature.feature_entry import FeatureEntry
from domains.feature.features_post_computers import FeaturesPostComputer, NoPostComputing


class KeepSpecified(FeaturesPostComputer):
    def __init__(self, labels_to_keep: list[str]) -> None:
        super().__init__()
        self.labels_to_keep = labels_to_keep

    def compute(self, features: list[dict[str, FeatureEntry]]) -> list[dict[str, float]]:
        return NoPostComputing().compute(
            [
                dict(
                    # (feature_key, feature_value)
                    # for feature_key, feature_value in feature.items()
                    # if feature_key in self.labels_to_keep
                    (label_to_keep, feature.get(label_to_keep, FeatureEntry(0.0, 0)))
                    for label_to_keep in self.labels_to_keep
                )
                for feature in features
            ]
        )
