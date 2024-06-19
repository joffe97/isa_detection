import logging
from typing import Optional, Type
from pandas import DataFrame
import numpy as np
from domains.caching.cache_func_decorator import cache_func

from domains.feature.feature_computer_container import FeatureComputerContainer
from domains.feature.features_post_computers.features_post_computer import (
    FeaturesPostComputer,
)
from domains.feature.features_post_computers.keep_specified import KeepSpecified
from domains.label.label_entry import LabelEntry
from domains.label.label_loaders.corpus_labels import CorpusLabels
from domains.label.label_loaders.label_loader import LabelLoader


class FeatureComputerContainerCollection:
    def __init__(
        self, feature_computer_containers: list[FeatureComputerContainer]
    ) -> None:
        self.feature_computer_containers = sorted(
            feature_computer_containers,
            key=lambda container: container.identifier(),
        )

    def identifiers(
        self,
        ignore_features_post_computers: Optional[
            list[Type[FeaturesPostComputer]]
        ] = None,
    ) -> list[str]:
        if ignore_features_post_computers is None:
            ignore_features_post_computers = []

        return [
            feature_computer_container.identifier(
                ignore_features_post_computers
            )
            for feature_computer_container in self.feature_computer_containers
        ]

    def identifier(
        self,
        seperator="_",
        ignore_features_post_computers: Optional[
            list[Type[FeaturesPostComputer]]
        ] = None,
    ) -> str:
        if ignore_features_post_computers is None:
            ignore_features_post_computers = []

        return seperator.join(self.identifiers(ignore_features_post_computers))

    def compute_for_binary_files(  # TODO: Refactor this method
        self,
        architecture_binary_files_dict: dict[str, list[str]],
        label_loader: LabelLoader,
    ) -> tuple[DataFrame, int]:
        binary_file_labels_list = (
            label_loader.load_as_binary_file_labels_mapping(
                architecture_binary_files_dict
            )
        )

        included_labels = binary_file_labels_list[0][1].included_labels()
        logging.debug(f"Included labels: {included_labels}")

        binary_files_features_mapping = dict()
        for i, feature_computer_container in enumerate(
            self.feature_computer_containers
        ):
            is_last_feature_computer_container = i == (
                len(self.feature_computer_containers) - 1
            )
            binary_file_list, labels_list = tuple(
                zip(*(binary_file_labels_list))
            )

            cur_computer_arch_features_list = (
                feature_computer_container.compute(binary_file_list)
            )

            for binary_file, arch_features, labels in zip(
                binary_file_list, cur_computer_arch_features_list, labels_list
            ):
                if is_last_feature_computer_container:
                    arch_features.update(
                        (key.value, value) for key, value in labels.items()
                    )
                binary_files_features_mapping.setdefault(binary_file, dict())
                binary_files_features_mapping[binary_file].update(arch_features)

        features = list(binary_files_features_mapping.values())
        feature_count = max(
            len(feature_dict) for feature_dict in features
        ) - len(included_labels)

        features_dataframe = DataFrame.from_records(features).fillna(0.0)

        return features_dataframe, feature_count

    # # @cache_func(use_class_identifier_method=True)
    # def compute_for_binary_files(  # TODO: Refactor this method
    #     self, architecture_binary_files_dict: dict[str, list[str]], label_loader: LabelLoader
    # ) -> tuple[DataFrame, int]:
    #     binary_file_labels_list = label_loader.load_as_binary_file_labels_mapping(
    #         architecture_binary_files_dict
    #     )

    #     included_labels = binary_file_labels_list[0][1].included_labels()
    #     logging.debug(f"Included labels: {included_labels}")

    #     data_frame_values = []
    #     data_frame_columns = []

    #     binary_files_features_mapping = dict()
    #     for i, feature_computer_container in enumerate(self.feature_computer_containers):
    #         is_last_feature_computer_container = i == (len(self.feature_computer_containers) - 1)
    #         binary_file_list, labels_list = tuple(zip(*(binary_file_labels_list)))

    #         cur_computer_arch_features_list = feature_computer_container.compute(binary_file_list)
    #         added_data_frame_columns = False
    #         for binary_file_index, (binary_file, arch_features, labels) in enumerate(
    #             zip(binary_file_list, cur_computer_arch_features_list, labels_list)
    #         ):
    #             if is_last_feature_computer_container:
    #                 arch_features.update((key.value, value) for key, value in labels.items())
    #             if not added_data_frame_columns:
    #                 added_data_frame_columns = True
    #                 data_frame_columns.extend(list(arch_features.keys()))

    #             # np_feature_values = np.fromiter(arch_features.values(), dtype=np.float)
    #             feature_values = list(arch_features.values())
    #             if len(data_frame_values) >= binary_file_index:
    #                 data_frame_values.append([feature_values])
    #             else:
    #                 data_frame_values[binary_file_index].append(feature_values)

    #             # binary_files_features_mapping.setdefault(binary_file, dict())
    #             # binary_files_features_mapping[binary_file].update(arch_features)

    #     # features = list(binary_files_features_mapping.values())
    #     # feature_count = max(len(feature_dict) for feature_dict in features) - len(included_labels)
    #     feature_count = len(data_frame_columns) - len(included_labels)

    #     features_dataframe = DataFrame.from_records(
    #         [
    #             [item for row in binary_file_feature_lists for item in row]
    #             for binary_file_feature_lists in data_frame_values
    #         ],
    #         columns=data_frame_columns,
    #     )
    #     return features_dataframe, feature_count
