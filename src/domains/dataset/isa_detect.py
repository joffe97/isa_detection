from glob import glob
from typing import Optional

from config import Config
from domains.dataset.binary_file_dataset import BinaryFileDataset


class IsaDetect(BinaryFileDataset):
    def __init__(self, file_index_end: Optional[int] = -1) -> None:
        super().__init__()
        self.file_index_end = file_index_end

    def create_architecture_paths_mapping(self) -> dict[str, list[str]]:
        architecture_binary_files_dict = dict(
            (
                arch_dir.split("/")[-1],
                (
                    file_paths := sorted(
                        file_path for file_path in glob(f"{arch_dir}/*") if not file_path.endswith(".json")
                    )
                )[: self.file_index_end or len(file_paths)],
            )
            for arch_dir in glob(f"{Config.ISA_DETECT_DATASET_PATH}/new_new_dataset/binaries/*")
        )
        return architecture_binary_files_dict
