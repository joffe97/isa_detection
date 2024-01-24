
from sklearn.base import ClassifierMixin

from domains.feature.feature_computer_collection import FeatureComputerCollection
from domains.feature.isa_binary_features_picker import ISABinaryFeaturesPicker
from domains.model.isa_model_collection import ISAModelCollection
from .system_mode import SystemMode


class TestModel(SystemMode):
    def run(self, feature_computer_collection: FeatureComputerCollection, classifier: ClassifierMixin, files_per_architecture: int) -> float:
        isa_binary_features = ISABinaryFeaturesPicker(
            feature_computer_collection).isadetect_features_full_binaries(files_per_architecture)
        isa_model_collection = ISAModelCollection(classifier).with_isa_binary_features(
            isa_binary_features)
        isa_model = isa_model_collection.isa_models[1]
        isa_model.train()
        classifier_fitted = isa_model.classifier

        cpu_rec_features = ISABinaryFeaturesPicker(
            feature_computer_collection).cpu_rec_corpus_features()
        cpu_rec_model_collection = ISAModelCollection(
            classifier_fitted).with_isa_binary_features(cpu_rec_features, clone_classifier=False)

        precision = cpu_rec_model_collection.mean_precision()

        cpu_rec_model_collection.print_precisions()
        return precision
