from domains.caching.cache_func_decorator import cache_func
from domains.feature.features_post_computers import FeaturesPostComputer


class MostCommon(FeaturesPostComputer):
    def __init__(self, n: int) -> None:
        self.n = n

    @cache_func()
    def compute(self, features: list[dict[str, float]]) -> list[dict[str, float]]:
        first_features = next(iter(features), None)
        if first_features is None:
            return []

        total_counts_dict = dict()
        for feature in features:
            for feature_key, feature_value in feature.items():
                total_counts_dict.setdefault(feature_key, 0.0)
                total_counts_dict[feature_key] += feature_value

        total_counts_items = sorted(total_counts_dict.items(
        ), key=lambda key_value: key_value[1], reverse=True)

        most_common_items = total_counts_items[:self.n]
        most_common_keys = set(key for key, _ in most_common_items)
        most_common_keys_kept_order = list(filter(
            lambda key: key in most_common_keys, first_features.keys()))

        most_common_features = [dict(
            (most_common_key, feature.get(most_common_key, 0.0)) for most_common_key in most_common_keys_kept_order) for feature in features]

        return most_common_features
