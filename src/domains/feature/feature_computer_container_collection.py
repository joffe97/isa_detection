import logging
from pathlib import Path
from pandas import DataFrame

from domains.feature.feature_computer_container import FeatureComputerContainer


class FeatureComputerContainerCollection:
    def __init__(self, feature_computer_containers: list[FeatureComputerContainer]) -> None:
        self.feature_computer_containers = feature_computer_containers

    def identifiers(self) -> list[str]:
        return [feature_computer_container.identifier() for feature_computer_container in self.feature_computer_containers]

    def identifier(self, seperator="_") -> str:
        return seperator.join(self.identifiers())

    def compute_for_binary_files(self, architecture_binary_files_dict: dict[str, list[str]], architecture_labels_mapping: dict[str, list[str]]) -> tuple[DataFrame, int]:
        architecture_binary_files_dict_items = sorted(
            architecture_binary_files_dict.items(), key=lambda item: item[0])

        binary_file_labels_list: list[tuple[str, dict[str, object]]] = []
        included_labels = set()
        architecture_count = 0
        for architecture_text, binary_files in architecture_binary_files_dict_items:
            labels = architecture_labels_mapping.get(architecture_text).copy()
            if labels is None:
                continue

            architecture_count += 1
            labels["architecture"] = architecture_count

            if len(included_labels) == 0:
                included_labels = set(labels)

            for include_label in list(included_labels):
                if include_label not in labels:
                    included_labels.remove(include_label)
                    for _, other_labels in binary_file_labels_list:
                        other_labels.pop(include_label, None)
                for label in list(labels.keys()):
                    if label not in included_labels:
                        labels.pop(label, None)

            for binary_file in binary_files:
                binary_file_labels_list.append((binary_file, labels))

        logging.debug(f"Included labels: {included_labels}")

        binary_files_features_mapping = dict()
        for i, feature_computer_container in enumerate(self.feature_computer_containers):
            is_last_feature_computer_container = (
                i == (len(self.feature_computer_containers) - 1))
            feature_computer_results = []

            binary_file_list, labels_list = tuple(
                zip(*(binary_file_labels_list)))
            for binary_file in binary_file_list:
                feature_computer_result = feature_computer_container.file_feature_computer.compute(
                    binary_file)
                feature_computer_results.append(feature_computer_result)

            cur_computer_arch_features_list = feature_computer_container.features_post_computer.compute(
                feature_computer_results)
            for binary_file, arch_features, labels in zip(binary_file_list, cur_computer_arch_features_list, labels_list):
                if is_last_feature_computer_container:
                    arch_features.update(labels)
                binary_files_features_mapping.setdefault(binary_file, dict())
                binary_files_features_mapping[binary_file].update(
                    arch_features)

        features = list(binary_files_features_mapping.values())

        feature_count = max(len(feature_dict)
                            for feature_dict in features) - len(included_labels)

        features_dataframe = DataFrame.from_records(features)
        return features_dataframe, feature_count
