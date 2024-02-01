import sklearn
from sklearn.base import ClassifierMixin
from domains.feature.feature_computer_collection import FeatureComputerCollection


class ISAModelConfiguration:
    def __init__(self, feature_computer_collection: FeatureComputerCollection, classifier: ClassifierMixin, files_per_architecture: int, target_label: str) -> None:
        self.feature_computer_collection = feature_computer_collection
        self.classifier = classifier
        self.files_per_architecture = files_per_architecture
        self.target_label = target_label

    def create_every_combination(feature_computer_collections: list[FeatureComputerCollection], classifiers: list[ClassifierMixin], files_per_architecture_list: list[int], target_labels: list[str]) -> list["ISAModelConfiguration"]:
        return [ISAModelConfiguration(feature_computer_collection, sklearn.base.clone(classifier), files_per_architecture, target_label)
                for target_label in target_labels
                for classifier in classifiers
                for feature_computer_collection in feature_computer_collections
                for files_per_architecture in files_per_architecture_list
                ]

    def __str__(self):
        return f"{self.feature_computer_collection.get_feature_computer_strs()}\n{self.classifier}\nFPA={self.files_per_architecture}\nTarget={self.target_label}"
