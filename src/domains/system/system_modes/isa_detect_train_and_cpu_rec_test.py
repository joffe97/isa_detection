
from domains.feature.isa_binary_features_picker import ISABinaryFeaturesPicker
from domains.model.isa_model_collection import ISAModelCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.model.info.isa_model_result_collection import ISAModelResultCollection
from .system_mode import SystemMode


class ISADetectTrainAndCpuRecTest(SystemMode):
    def run(self, isa_model_configuration: ISAModelConfiguration) -> ISAModelResultCollection:
        isa_binary_features = ISABinaryFeaturesPicker(
            isa_model_configuration.feature_computer_collection, isa_model_configuration.target_label).isadetect_features_full_binaries(isa_model_configuration.files_per_architecture)
        isa_model_collection = ISAModelCollection(isa_model_configuration.classifier).with_isa_binary_features(
            isa_binary_features)
        isa_model = isa_model_collection.isa_models[1]
        isa_model.train()
        classifier_fitted = isa_model.classifier

        cpu_rec_features = ISABinaryFeaturesPicker(
            isa_model_configuration.feature_computer_collection, isa_model_configuration.target_label).cpu_rec_corpus_features()
        cpu_rec_model_collection = ISAModelCollection(
            classifier_fitted).with_isa_binary_features(cpu_rec_features, clone_classifier=False)

        results = cpu_rec_model_collection.find_results()

        # cpu_rec_model_collection.print_endianness_precisions()
        return results
