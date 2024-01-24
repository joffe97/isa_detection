from sklearn.base import ClassifierMixin
import sklearn
from pandas import DataFrame

from domains.feature.feature_computer_collection import FeatureComputerCollection
from domains.system.system_modes import SystemMode


class System():
    def __init__(self, system_mode: SystemMode, feature_computer_collections: list[FeatureComputerCollection], classifiers: list[ClassifierMixin], files_per_architecture_list: list[int]) -> None:
        self.system_mode = system_mode
        self.feature_computer_collections = feature_computer_collections
        self.classifiers = classifiers
        self.files_per_architecture_list = files_per_architecture_list

    def create_model_combinations(self) -> list[tuple[FeatureComputerCollection, ClassifierMixin, int]]:
        return [(feature_computer_collection, sklearn.base.clone(classifier), files_per_architecture)
                for classifier in self.classifiers for feature_computer_collection in self.feature_computer_collections for files_per_architecture in self.files_per_architecture_list]

    def get_precisions(self) -> tuple[list[str], ClassifierMixin, int, float]:
        combinations = self.create_model_combinations()

        results = []
        for (feature_computer_collection, classifier, files_per_architecture) in combinations:
            precision = self.system_mode.run(
                feature_computer_collection,
                classifier,
                files_per_architecture)

            feature_computer_method_strs = feature_computer_collection.get_feature_computer_strs()
            results.append(
                (feature_computer_method_strs, classifier, files_per_architecture, precision))
            print(
                f"{feature_computer_method_strs}\n{classifier}\nFPA={files_per_architecture}\nPrecision: {precision:.4f}\n\n")
        return results

    def run(self) -> DataFrame:
        precisions = self.get_precisions()

        precision_classifier_dict = dict()
        features = []
        for item in precisions:
            features_str = " + ".join(item[0])
            classifier_str = str(item[1])
            files_per_architecture = item[2]
            row_str = f"{classifier_str}, (FPA={files_per_architecture})"
            precision = item[3]
            precision_classifier_dict.setdefault(row_str, [])
            precision_classifier_dict[row_str].append(precision)
            if features_str not in features:
                features.append(features_str)

        return DataFrame.from_dict(precision_classifier_dict, columns=features, orient="index")
