import json
from pandas import DataFrame, Series
import xxhash

from config import Config
from domains.caching import Caching
from domains.feature.feature_computer_container_collection import (
    FeatureComputerContainerCollection,
)
from domains.label.label_entry import LabelEntry
from domains.label.label_loaders.corpus_labels import CorpusLabels


class ISABinaryFeatures:
    def __init__(
        self,
        identifier: str,
        dataframe: DataFrame,
        target_label: LabelEntry,
        feature_count: int,
    ) -> None:
        self.identifier = identifier
        self.dataframe = dataframe
        self.target_label = target_label
        self.feature_count = feature_count

        if self.target_label.value not in self.dataframe.columns:
            raise KeyError(f"target_label does not exist in dataframe: {self.target_label.value}")

    @property
    def data(self) -> DataFrame:
        return self.dataframe.iloc[:, 0 : self.feature_count]

    @property
    def target(self) -> Series:
        return self.get_column(self.target_label.value)

    @property
    def architecture_ids(self) -> Series:
        return self.get_column("architecture")

    @property
    def architecture_texts(self) -> Series:
        return self.get_column("architecture_text")

    def get_column(self, column_name: str) -> Series:
        return self.dataframe[column_name]

    # def pretty_print(self) -> None:
    #     print("Identifier:")
    #     # ipython_display(self.identifier)
    #     print()
    #     print("Feature_count:")
    #     # ipython_display(self.feature_count)
    #     print()
    #     print("Dataframe:")
    #     # ipython_display(self.dataframe)

    # def _add_labels_to_data(self, labels: list[dict]) -> DataFrame:
    #     def create_column(column_name: str):
    #         column_repeated_gen = (
    #             np.repeat(
    #                 column_label, len(self.dataframe[self.architecture_ids == i + 1])
    #             )
    #             for i, column_label in enumerate(
    #                 list(map(lambda label: label[column_name], labels))
    #             )
    #         )
    #         return [
    #             column
    #             for column_repeated in column_repeated_gen
    #             for column in column_repeated
    #         ]

    #     # Create wordsize column
    #     wordsize_column = create_column("wordsize")

    #     # Create endianness column
    #     endianness_column = create_column("endianness")

    #     # Create architecture text column
    #     arch_text_column = create_column("architecture_text")

    #     # Add columns to dataframe
    #     self.dataframe["wordsize"] = wordsize_column
    #     self.dataframe["endianness"] = endianness_column
    #     self.dataframe["architecture_text"] = arch_text_column

    #     return self.dataframe

    # Load data from ISAdetect binary features csv
    # @staticmethod
    # def load_from_isadetect_features_csv(csv_filename: str) -> "ISABinaryFeatures":
    #     identifier = ".".join(csv_filename.split("/")[-1].split(".")[:-1]).lower()

    #     df_full = pandas.read_csv(csv_filename)
    #     df_full.rename(
    #         columns={
    #             "be_one": "bigram_0x0001",
    #             "be_stack": "bigram_0xfffe",
    #             "le_one": "bigram_0x0100",
    #             "le_stack": "bigram_0xfeff",
    #         },
    #         inplace=True,
    #     )

    #     # Remove unwanted columns
    #     unwanted_columns = [
    #         "amd64_epilog_1",
    #         "amd64_epilog_2",
    #         "amd64_epilog_3",
    #         "amd64_prolog_1",
    #         "amd64_prolog_2",
    #         "arm32_epilog_1",
    #         "arm32_epilog_2",
    #         "arm32_prolog_1",
    #         "arm32_prolog_2",
    #         "armel32_epilog_1",
    #         "armel32_epilog_2",
    #         "armel32_prolog_1",
    #         "armel32_prolog_2",
    #         "mips32_epilog_1",
    #         "mips32_prolog_1",
    #         "mips32_prolog_2",
    #         "mips32el_epilog_1",
    #         "mips32el_prolog_1",
    #         "mips32el_prolog_2",
    #         "ppc32_epilog_1",
    #         "ppc32_prolog_1",
    #         "ppc64_epilog_1",
    #         "ppc64_prolog_1",
    #         "ppc64_prolog_2",
    #         "ppc64_prolog_3",
    #         "ppcel32_epilog_1",
    #         "ppcel32_prolog_1",
    #         "ppcel64_epilog_1",
    #         "ppcel64_prolog_1",
    #         "s390x_epilog_1",
    #         "s390x_prolog_1",
    #         "powerpcspe_spe_instruction_evl",
    #         "powerpcspe_spe_instruction_isel",
    #     ]
    #     df_full = df_full.drop(columns=unwanted_columns)

    #     isa_binary_features = ISABinaryFeatures(identifier, df_full, 256 + 4)
    #     isa_binary_features._add_labels_to_data(Labels.get_isa_detect_csv_labels())

    #     return isa_binary_features

    @classmethod
    def load_or_create_from_binary_files(
        cls,
        architecture_binary_files_dict: dict[str, list[str]],
        feature_computer_container_collection: FeatureComputerContainerCollection,
        target_label: LabelEntry,
    ) -> "ISABinaryFeatures":
        for binary_files in architecture_binary_files_dict.values():
            binary_files.sort()
        architecture_binary_files_json = json.dumps(architecture_binary_files_dict, sort_keys=True)
        architecture_binary_files_hash = xxhash.xxh32(architecture_binary_files_json).hexdigest()

        binary_file_feature_computer_str = feature_computer_container_collection.identifier()

        identifier = "/".join(
            [
                target_label.value,
                binary_file_feature_computer_str,
                architecture_binary_files_hash,
            ]
        )
        isa_binary_feature_path = Config.CACHE_PATH.joinpath("isa_binary_features", identifier)

        return Caching().load_or_process_func_data(
            isa_binary_feature_path,
            lambda: cls.from_binary_files(
                identifier,
                architecture_binary_files_dict,
                feature_computer_container_collection,
                target_label,
            ),
        )

    @staticmethod
    def from_binary_files(
        identifier: str,
        architecture_binary_files_dict: dict[str, list[str]],
        feature_computer_container_collection: FeatureComputerContainerCollection,
        target_label: LabelEntry,
    ) -> "ISABinaryFeatures":
        label_loader = CorpusLabels.with_default_included_labels({target_label})
        features_dataframe, features_count = feature_computer_container_collection.compute_for_binary_files(
            architecture_binary_files_dict, label_loader
        )

        return ISABinaryFeatures(identifier, features_dataframe, target_label, features_count)

    # def with_n_most_common_column_group(
    #     self, n: int, column_group: str
    # ) -> "ISABinaryFeatures":
    #     data_mean_column_groups = self.dataframe.filter(like=column_group).mean()
    #     data_mean_column_groups.sort_values(ascending=False, inplace=True)
    #     data_mean_column_groups_least_common = data_mean_column_groups[n:]

    #     self.dataframe.drop(
    #         columns=data_mean_column_groups_least_common.index.to_list(), inplace=True
    #     )

    #     self.feature_count -= len(data_mean_column_groups_least_common)
    #     return self

    # def with_n_most_common_n_grams_endianness_dependent(
    #     self, n: int, n_gram_bytes: int
    # ) -> "ISABinaryFeatures":
    #     if n_gram_bytes == 2:
    #         n_gram_name = "bigram"
    #     elif n_gram_bytes == 3:
    #         n_gram_name = "trigram"
    #     else:
    #         raise NotImplementedError(
    #             f"case is not implemented for n_gram_name with the following byte count: {n_gram_bytes}"
    #         )

    #     data_mean_little_endianness = (
    #         self.dataframe[self.dataframe["endianness"] == "Little"]
    #         .filter(like=n_gram_name)
    #         .mean()
    #     )
    #     data_mean_big_endianness = (
    #         self.dataframe[self.dataframe["endianness"] == "Big"]
    #         .filter(like=n_gram_name)
    #         .mean()
    #     )

    #     bigram_value_count = Series(np.zeros(int("0xffff", 16) + 1))

    #     for label, frequency in data_mean_little_endianness.items():
    #         bigram = label.split("_")[-1]
    #         bigram_be = int(bigram, 16)
    #         bigram_le = (bigram_be >> 8) | ((bigram_be & int("0xff", 16)) << 8)
    #         bigram_value_count[bigram_le] += frequency

    #     for label, frequency in data_mean_big_endianness.items():
    #         bigram = label.split("_")[-1]
    #         bigram_be = int(bigram, 16)
    #         bigram_value_count[bigram_be] += frequency

    #     bigram_value_count.sort_values(ascending=False, inplace=True)
    #     bigram_values_least_common = bigram_value_count[n // 2 :].index.to_list()

    #     bigram_labels_to_drop = set()
    #     for bigram_value in bigram_values_least_common:
    #         bigram_be = f"0x{hex(bigram_value)[2:].zfill(4)}"
    #         bigram_le = f"0x{bigram_be[4:6]}{bigram_be[2:4]}"
    #         bigram_labels_to_drop.update(
    #             f"{n_gram_name}_{bigram}" for bigram in [bigram_be, bigram_le]
    #         )

    #     self.dataframe.drop(columns=bigram_labels_to_drop, inplace=True)

    #     self.feature_count -= len(bigram_labels_to_drop)
    #     return self
