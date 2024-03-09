import logging
from pandas import DataFrame
from domains.caching.cache_func_decorator import cache_func

from domains.feature.feature_computer_container import FeatureComputerContainer
from domains.label.label_entry import LabelEntry
from domains.label.label_loaders.corpus_labels import CorpusLabels
from domains.label.label_loaders.label_loader import LabelLoader


class FeatureComputerContainerCollection:
    def __init__(self, feature_computer_containers: list[FeatureComputerContainer]) -> None:
        self.feature_computer_containers = sorted(
            feature_computer_containers, key=lambda container: container.identifier()
        )

    def identifiers(self) -> list[str]:
        return [
            feature_computer_container.identifier()
            for feature_computer_container in self.feature_computer_containers
        ]

    def identifier(self, seperator="_") -> str:
        return seperator.join(self.identifiers())

    # @cache_func(use_class_identifier_method=True)
    def compute_for_binary_files(  # TODO: Refactor this method
        self, architecture_binary_files_dict: dict[str, list[str]], label_loader: LabelLoader
    ) -> tuple[DataFrame, int]:
        binary_file_labels_list = label_loader.load_as_binary_file_labels_mapping(
            architecture_binary_files_dict
        )

        included_labels = binary_file_labels_list[0][1].included_labels()
        logging.debug(f"Included labels: {included_labels}")

        binary_files_features_mapping = dict()
        for i, feature_computer_container in enumerate(self.feature_computer_containers):
            is_last_feature_computer_container = i == (len(self.feature_computer_containers) - 1)
            binary_file_list, labels_list = tuple(zip(*(binary_file_labels_list)))

            cur_computer_arch_features_list = feature_computer_container.compute(binary_file_list)

            for binary_file, arch_features, labels in zip(
                binary_file_list, cur_computer_arch_features_list, labels_list
            ):
                if is_last_feature_computer_container:
                    arch_features.update((key.value, value) for key, value in labels.items())
                binary_files_features_mapping.setdefault(binary_file, dict())
                binary_files_features_mapping[binary_file].update(arch_features)

        features = list(binary_files_features_mapping.values())
        feature_count = max(len(feature_dict) for feature_dict in features) - len(included_labels)

        features_dataframe = DataFrame.from_records(features)
        return features_dataframe, feature_count
