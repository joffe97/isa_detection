import sklearn
from sklearn.base import (
    BaseEstimator,
)
from domains.feature.feature_computer_container_collection import (
    FeatureComputerContainerCollection,
)
from domains.feature.features_post_computers import (
    FeaturesPostComputer,
)
from domains.label.label_entry import LabelEntry


class ISAModelConfiguration:
    def __init__(
        self,
        feature_computer_container_collection: FeatureComputerContainerCollection,
        classifier: BaseEstimator,
        files_per_architecture: int,
        target_label: LabelEntry,
    ) -> None:
        self.feature_computer_container_collection = feature_computer_container_collection
        self.classifier = classifier
        self.files_per_architecture = files_per_architecture
        self.target_label = target_label

    @staticmethod
    def create_every_combination(
        feature_computer_container_collections: list[FeatureComputerContainerCollection],
        classifiers: list[BaseEstimator],
        files_per_architecture_list: list[int],
        target_labels: list[LabelEntry],
    ) -> list["ISAModelConfiguration"]:
        return [
            ISAModelConfiguration(
                feature_computer_container_collection,
                sklearn.base.clone(classifier),
                files_per_architecture,
                target_label,
            )
            for target_label in target_labels
            for classifier in classifiers
            for feature_computer_container_collection in feature_computer_container_collections
            for files_per_architecture in files_per_architecture_list
        ]

    def change_all_features_post_computers(
        self,
        new_features_post_computer: FeaturesPostComputer,
    ) -> None:
        for (
            feature_computer_container
        ) in self.feature_computer_container_collection.feature_computer_containers:
            feature_computer_container.features_post_computer = new_features_post_computer

    def __str__(
        self,
    ):
        return f"{self.feature_computer_container_collection.identifier()}\n{self.classifier}\nFPA={self.files_per_architecture}\nTarget={self.target_label}"
