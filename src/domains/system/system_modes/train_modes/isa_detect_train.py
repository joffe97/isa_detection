from domains.feature.isa_binary_features_picker import ISABinaryFeaturesPicker
from domains.model.isa_model_collection import ISAModelCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.system_modes.train_modes.train_mode import TrainMode


class ISADetectTrain(TrainMode):
    def run(isa_model_configuration: ISAModelConfiguration) -> ISAModelCollection:
        isa_binary_features = ISABinaryFeaturesPicker(
            isa_model_configuration.feature_computer_collection, isa_model_configuration.target_label).isadetect_features_full_binaries(isa_model_configuration.files_per_architecture)
        return ISAModelCollection(isa_model_configuration.classifier).with_isa_binary_features(
            isa_binary_features)
