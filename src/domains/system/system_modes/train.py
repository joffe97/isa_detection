from sklearn.base import ClassifierMixin

from domains.feature.feature_computer_collection import FeatureComputerCollection
from domains.feature.isa_binary_features_picker import ISABinaryFeaturesPicker
from domains.model.isa_model_collection import ISAModelCollection
from .system_mode import SystemMode


class Train(SystemMode):
    def run(self, feature_computer_collection: FeatureComputerCollection, classifier: ClassifierMixin, files_per_architecture: int) -> float:
        isa_binary_features = ISABinaryFeaturesPicker(
            feature_computer_collection).isadetect_features_full_binaries(files_per_architecture)

        isa_model_collection = ISAModelCollection(classifier).with_isa_binary_features(
            isa_binary_features)

        precision = isa_model_collection.mean_precision()

        isa_model_collection.print_precisions()
        return precision
