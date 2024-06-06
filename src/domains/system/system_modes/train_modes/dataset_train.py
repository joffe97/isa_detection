from typing import Optional, Type
from domains.dataset.binary_file_dataset import BinaryFileDataset
from domains.feature.isa_binary_features_picker import ISABinaryFeaturesPicker
from domains.model.isa_model_collection import ISAModelCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.system_modes.train_modes.train_mode import TrainMode


class DatasetTrain(TrainMode):
    def __init__(self, dataset_class: Type[BinaryFileDataset]) -> None:
        super().__init__()
        self.dataset_class = dataset_class

    def get_dataset(self, files_per_architecture: Optional[int]) -> BinaryFileDataset:
        try:
            dataset = self.dataset_class(files_per_architecture)  # type: ignore
        except TypeError:
            dataset = self.dataset_class()
        return dataset

    def run(self, isa_model_configuration: ISAModelConfiguration) -> ISAModelCollection:
        isa_binary_features = ISABinaryFeaturesPicker(
            isa_model_configuration.feature_computer_container_collection,
            isa_model_configuration.target_label,
        ).binary_file_dataset_features(self.get_dataset(isa_model_configuration.files_per_architecture))
        return ISAModelCollection(isa_model_configuration.classifier).with_isa_binary_features(
            isa_binary_features
        )
