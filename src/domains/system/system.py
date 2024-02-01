from pandas import DataFrame

from domains.model.info.isa_model_info import ISAModelInfo
from domains.model.info.isa_model_info_collection import ISAModelInfoCollection
from domains.model.isa_model_configuration import ISAModelConfiguration
from domains.system.system_modes import SystemMode


class System():
    def __init__(self, system_mode: SystemMode, isa_model_configurations: list[ISAModelConfiguration]) -> None:
        self.system_mode = system_mode
        self.isa_model_configurations = isa_model_configurations

    def get_isa_model_info_collection(self) -> ISAModelInfoCollection:
        isa_model_info_list = []
        for isa_model_configuration in self.isa_model_configurations:
            result_collection = self.system_mode.run(isa_model_configuration)
            isa_model_info = ISAModelInfo(
                isa_model_configuration, result_collection)

            isa_model_info_list.append(isa_model_info)
        isa_model_info_collection = ISAModelInfoCollection(isa_model_info_list)
        isa_model_info_collection.print()
        return isa_model_info_collection

    def run(self) -> DataFrame:
        isa_model_info_collection = self.get_isa_model_info_collection()

        precision_classifier_dict = dict()
        features = []
        for isa_model_info in isa_model_info_collection.collection:
            features_str = " + ".join(
                isa_model_info.configuration.feature_computer_collection.get_feature_computer_strs())
            classifier_str = str(isa_model_info.configuration.classifier)
            files_per_architecture = isa_model_info.configuration.files_per_architecture
            target_label = isa_model_info.configuration.target_label
            row_str = f"{classifier_str}, (FPA={files_per_architecture}, target={target_label})"
            precision = isa_model_info.results.mean_precision()
            precision_classifier_dict.setdefault(row_str, [])
            precision_classifier_dict[row_str].append(precision)
            if features_str not in features:
                features.append(features_str)

        return DataFrame.from_dict(precision_classifier_dict, columns=features, orient="index")
