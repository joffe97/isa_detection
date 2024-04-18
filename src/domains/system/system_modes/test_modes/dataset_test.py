from domains.dataset.binary_file_dataset import BinaryFileDataset
from domains.dataset.cpu_rec import CpuRec
from domains.feature.features_post_computers import KeepSpecified
from domains.feature.isa_binary_features_picker import ISABinaryFeaturesPicker
from domains.model.info.isa_model_result_collection import ISAModelResultCollection
from domains.model.isa_model_collection import ISAModelCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.system_modes.test_modes.test_mode import TestMode


class DatasetTest(TestMode):
    def __init__(self, dataset: BinaryFileDataset) -> None:
        super().__init__()
        self.dataset = dataset

    def run(
        self,
        isa_model_configuration: ISAModelConfiguration,
        isa_model_collection: ISAModelCollection,
    ) -> ISAModelResultCollection:
        isa_model = isa_model_collection.isa_models[1]
        isa_model.train()
        classifier_fitted = isa_model.classifier

        labels = isa_model.get_labels()
        isa_model_configuration.change_all_features_post_computers(KeepSpecified(labels))

        dataset_features = ISABinaryFeaturesPicker(
            isa_model_configuration.feature_computer_container_collection,
            isa_model_configuration.target_label,
        ).binary_file_dataset_features(self.dataset)
        dataset_model_collection = ISAModelCollection(classifier_fitted).with_isa_binary_features(
            dataset_features, clone_classifier=False
        )

        return dataset_model_collection.find_results()
