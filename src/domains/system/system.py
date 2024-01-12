from sklearn.base import ClassifierMixin
import sklearn
from pandas import DataFrame, Series

from domains.feature.feature_computer_collection import FeatureComputerCollection
from domains.feature.isa_binary_features_picker import ISABinaryFeaturesPicker
from domains.feature.feature_computers import FeatureComputer
from domains.model.isa_model_collection import ISAModelCollection


class System():
    def __init__(self, feature_computer_collections: list[FeatureComputerCollection], classifiers: list[ClassifierMixin], files_per_architecture_list: list[int]) -> None:
        self.feature_computer_collections = feature_computer_collections
        self.classifiers = classifiers
        self.files_per_architecture_list = files_per_architecture_list

    def create_model_combinations(self) -> list[tuple[FeatureComputerCollection, ClassifierMixin, int]]:
        return [(feature_computer_collection, sklearn.base.clone(classifier), files_per_architecture)
                for classifier in self.classifiers for feature_computer_collection in self.feature_computer_collections for files_per_architecture in self.files_per_architecture_list]

    def run_with_features_and_classifier(self, binary_file_feature_computer: FeatureComputerCollection, classifier: object, files_per_architecture: int):
        isa_binary_features = ISABinaryFeaturesPicker(
            binary_file_feature_computer).isadetect_features_full_binaries(files_per_architecture)

        isa_model_collection = ISAModelCollection(classifier).with_isa_binary_features(
            isa_binary_features)

        precision = isa_model_collection.mean_precision()

        isa_model_collection.print_precisions()
        return precision

    def run_with_features_and_classifier_cpu_rec(self, binary_file_feature_computer: FeatureComputerCollection, classifier: object, files_per_architecture: int):
        isa_binary_features = ISABinaryFeaturesPicker(
            binary_file_feature_computer).isadetect_features_full_binaries(files_per_architecture)
        isa_model_collection = ISAModelCollection(classifier).with_isa_binary_features(
            isa_binary_features)
        isa_model = isa_model_collection.isa_models[1]
        isa_model.train()
        classifier_fitted = isa_model.classifier

        cpu_rec_features = ISABinaryFeaturesPicker(
            binary_file_feature_computer).cpu_rec_corpus_features()
        cpu_rec_model_collection = ISAModelCollection(
            classifier_fitted).with_isa_binary_features(cpu_rec_features, clone_classifier=False)

        precision = cpu_rec_model_collection.mean_precision()

        cpu_rec_model_collection.print_precisions()
        return precision

    def get_precisions(self) -> tuple[list[str], object, int, float]:
        combinations = self.create_model_combinations()

        results = []
        for (feature_computer_collection, classifier, files_per_architecture) in combinations:
            precision = self.run_with_features_and_classifier_cpu_rec(
                feature_computer_collection,
                classifier,
                files_per_architecture)

            feature_computer_method_strs = feature_computer_collection.get_compute_methods_strs()
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
