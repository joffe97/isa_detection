from domains.feature.feature_entry import FeatureEntry
from domains.feature.features_post_computers import FeaturesPostComputer


class MostCommon(FeaturesPostComputer):
    def __init__(self, n: int) -> None:
        self.n = n

    def compute(self, features: list[dict[str, FeatureEntry]]) -> list[dict[str, float]]:
        first_features = next(iter(features), None)
        if first_features is None:
            return []

        total_counts_dict: dict[str, FeatureEntry] = dict()
        for feature in features:
            for feature_key, feature_entry in feature.items():
                total_counts_dict.setdefault(
                    feature_key, FeatureEntry(0.0, feature_entry.numerical_identifier)
                )
                total_counts_dict[feature_key].value += feature_entry.value

        total_counts_items = sorted(
            total_counts_dict.items(),
            key=lambda key_value: key_value[1].value,
            reverse=True,
        )

        most_common_items = total_counts_items[: self.n]
        most_common_items_sorted = list(sorted(most_common_items, key=lambda item: item[1].value))
        most_common_keys_sorted = list(key for key, _ in most_common_items_sorted)

        most_common_features = [
            dict(
                (
                    most_common_key,
                    (
                        feature_entry.value
                        if (feature_entry := feature.get(most_common_key, None)) is not None
                        else 0.0
                    ),
                )
                for most_common_key in most_common_keys_sorted
            )
            for feature in features
        ]

        return most_common_features
