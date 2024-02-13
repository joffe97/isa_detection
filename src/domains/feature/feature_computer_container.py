from domains.feature.features_post_computers import (
    FeaturesPostComputer,
    NoPostComputing,
)
from domains.feature.file_feature_computer_collection import FileFeatureComputer


class FeatureComputerContainer:
    def __init__(
        self,
        file_feature_computer: FileFeatureComputer,
        features_post_computer: FeaturesPostComputer = NoPostComputing(),
    ) -> None:
        self.file_feature_computer = file_feature_computer
        self.features_post_computer = features_post_computer

    def identifier(self) -> str:
        feature_computers_str = self.file_feature_computer.__name__
        features_post_computer_str = (
            self.features_post_computer.identifier()
            if not isinstance(self.features_post_computer, NoPostComputing)
            else None
        )

        parts = list(filter(None, [feature_computers_str, features_post_computer_str]))
        return "+".join(parts)
