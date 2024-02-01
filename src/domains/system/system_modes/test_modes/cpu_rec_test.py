from domains.feature.isa_binary_features_picker import ISABinaryFeaturesPicker
from domains.model.info.isa_model_result_collection import ISAModelResultCollection
from domains.model.isa_model_collection import ISAModelCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.system_modes.test_modes.test_mode import TestMode


class CpuRecTest(TestMode):
    def run(isa_model_configuration: ISAModelConfiguration, isa_model_collection: ISAModelCollection) -> ISAModelResultCollection:
        isa_model = isa_model_collection.isa_models[1]
        isa_model.train()
        classifier_fitted = isa_model.classifier

        cpu_rec_features = ISABinaryFeaturesPicker(
            isa_model_configuration.feature_computer_collection, isa_model_configuration.target_label).cpu_rec_corpus_features()
        cpu_rec_model_collection = ISAModelCollection(
            classifier_fitted).with_isa_binary_features(cpu_rec_features, clone_classifier=False)

        results = cpu_rec_model_collection.find_results()
