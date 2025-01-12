from domains.dataset.binary_file_dataset import BinaryFileDataset
from domains.feature.feature_computer_container_collection import FeatureComputerContainerCollection
from domains.label.label_entry import LabelEntry
from .isa_binary_features import ISABinaryFeatures


class ISABinaryFeaturesPicker:
    def __init__(
        self,
        feature_computer_container_collection: FeatureComputerContainerCollection,
        target_label: LabelEntry,
    ) -> None:
        self.feature_computer_container_collection = feature_computer_container_collection
        self.target_label = target_label

    # @staticmethod
    # def isadetect_features_full_csv() -> ISABinaryFeatures:
    #     return ISABinaryFeatures.load_from_isadetect_features_csv(
    #         f"{Config.ISA_DETECT_DATASET_PATH}/new_new_dataset/ISAdetect_full_binaries_features.csv"
    #     )

    def binary_file_dataset_features(self, binary_file_dataset: BinaryFileDataset) -> "ISABinaryFeatures":
        return ISABinaryFeatures.load_or_create_from_binary_files(
            binary_file_dataset.create_architecture_paths_mapping(),
            self.feature_computer_container_collection,
            self.target_label,
        )

    # def isadetect_features_code_binaries(
    #     self, file_index_end: int = -1, identifier="isadetect_code_binaries"
    # ) -> ISABinaryFeatures:
    #     code_files = glob(f"{Config.ISA_DETECT_DATASET_PATH}/new_new_dataset/binaries_code_sections_only/*")
    #     architecture_binary_files_list = list(
    #         (
    #             glob(
    #                 f"{Config.ISA_DETECT_DATASET_PATH}/new_new_dataset/binaries/*/{code_file.split('/')[-1].split('.')[0]}"
    #             )[0].split("/")[-2],
    #             code_file,
    #         )
    #         for code_file in code_files
    #     )

    #     architecture_binary_files_dict = dict()
    #     architecture_file_index_end_dict = dict()
    #     for arch, file in architecture_binary_files_list:
    #         if (cur_file_index_end := architecture_file_index_end_dict.get(arch)) is None:
    #             if file_index_end < 0:
    #                 cur_file_index_end = (
    #                     len(
    #                         [cur_arch for (cur_arch, _) in architecture_binary_files_list if cur_arch == arch]
    #                     )
    #                     + file_index_end
    #                     + 1
    #                 )
    #             else:
    #                 cur_file_index_end = file_index_end
    #             architecture_file_index_end_dict[arch] = cur_file_index_end
    #         architecture_binary_files_dict.setdefault(arch, [])
    #         if len(architecture_binary_files_dict[arch]) < cur_file_index_end:
    #             architecture_binary_files_dict[arch].append(file)

    #     return ISABinaryFeatures.from_binary_files(
    #         identifier,
    #         architecture_binary_files_dict,
    #         self.feature_computer_container_collection,
    #         self.target_label,
    #     )

    # def binary_file_with_custom_labels(
    #     self, identifier: str, filename: str, labels: dict[str, object]
    # ) -> ISABinaryFeatures:
    #     architecture_text = labels.get("architecture_text", "unknown_architecture_text")
    #     architecture_binary_files_dict = {architecture_text: [filename]}
    #     labels_dict = {architecture_text: {"architecture_text": architecture_text, **labels}}
    #     return ISABinaryFeatures.from_binary_files(
    #         identifier,
    #         architecture_binary_files_dict,
    #         self.feature_computer_container_collection,
    #         labels_dict=labels_dict,
    #     )

    # def cpu_rec_file(self, identifier: str, filename: str) -> ISABinaryFeatures:
    #     isa = filename.split("/")[-1]
    #     labels_dict_tmp = dict((label["isa"], label) for label in Labels.get_corpus_labels())
    #     architecture_text = labels_dict_tmp.get(isa)["architecture_text"]

    #     if architecture_text is None:
    #         raise KeyError(f"Labels is not found for ISA: {isa}")
    #     elif not architecture_text:
    #         architecture_text = "unknown_architecture_text"

    #     architecture_binary_files_dict = {architecture_text: [filename]}
    #     return ISABinaryFeatures.from_binary_files(
    #         identifier, architecture_binary_files_dict, self.feature_computer_container_collection
    #     )
