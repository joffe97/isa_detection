from domains.feature.feature_entry import FeatureEntry
from domains.feature.features_post_computers import FeaturesPostComputer, NoPostComputing


class KeepSpecified(FeaturesPostComputer):
    def __init__(self, labels_to_keep: list[str]) -> None:
        super().__init__()
        self.labels_to_keep = labels_to_keep

    def _equals_labels_to_keep(self, labels: list[str]) -> bool:
        return labels == self.labels_to_keep

    def compute(self, features: list[dict[str, FeatureEntry]]) -> list[dict[str, float]]:
        if len(features) == 0:
            return []

        first_feature = features[0]
        feature_labels = list(first_feature.keys())

        if not self._equals_labels_to_keep(feature_labels):
            features = [
                (
                    dict(
                        (label_to_keep, feature.get(label_to_keep, FeatureEntry(0.0, 0)))
                        for label_to_keep in self.labels_to_keep
                    )
                )
                for feature in features
            ]

        return NoPostComputing().compute(features)
