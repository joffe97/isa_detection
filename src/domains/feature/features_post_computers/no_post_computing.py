from domains.feature.features_post_computers.features_post_computer import FeaturesPostComputer


class NoPostComputing(FeaturesPostComputer):
    def compute(self, features: list[dict[str, float]]) -> list[dict[str, float]]:
        return features
