from glob import glob

from config import Config
from domains.dataset.binary_file_dataset import BinaryFileDataset


class IsaDetect(BinaryFileDataset):
    def __init__(self, file_index_end: int = -1) -> None:
        super().__init__()
        self.file_index_end = file_index_end

    def create_architecture_paths_mapping(self) -> dict[str, list[str]]:
        architecture_binary_files_dict = dict(
            (arch_dir.split("/")[-1], glob(f"{arch_dir}/*")[: self.file_index_end])
            for arch_dir in glob(f"{Config.ISA_DETECT_DATASET_PATH}/new_new_dataset/binaries/*")
        )
        return architecture_binary_files_dict
