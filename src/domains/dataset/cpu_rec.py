from glob import glob

from config import Config
from domains.dataset.binary_file_dataset import BinaryFileDataset


class CpuRec(BinaryFileDataset):
    def create_architecture_paths_mapping(self) -> dict[str, list[str]]:
        # labels_dict_tmp = dict((label["isa"], label) for label in Labels.get_corpus_exclusive())
        architecture_binary_files_dict: dict[str, list[str]] = dict(
            (architecture_text, [path])
            for path in glob(f"{Config.CPU_REC_DATASET_PATH}/*.corpus")
            if (architecture_text := path.split("/")[-1]) and not architecture_text.startswith("_")
            # if (labels := labels_dict_tmp.get(path.split("/")[-1])) is not None
            # and (architecture_text := labels["architecture_text"])
        )  # type: ignore
        return architecture_binary_files_dict
