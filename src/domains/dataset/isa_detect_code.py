from glob import glob
from pathlib import Path
from typing import Optional

from config import Config
from domains.dataset.binary_file_dataset import BinaryFileDataset


class IsaDetectCode(BinaryFileDataset):
    def __init__(self, file_index_end: Optional[int] = None) -> None:
        super().__init__()
        self.file_index_end = file_index_end

    def create_architecture_paths_mapping(self) -> dict[str, list[str]]:
        architecture_binary_files_dict = dict(
            (
                arch_dir.split("/")[-1],
                (
                    file_paths := sorted(
                        file_path for file_path in glob(f"{arch_dir}/*") if file_path.endswith(".code")
                    )
                )[: self.file_index_end or len(file_paths)],
            )
            for arch_dir in glob(
                f"{Config.ISA_DETECT_DATASET_PATH}/new_new_dataset/binaries_code_sections_only/*"
            )
            if Path(arch_dir).is_dir()
        )
        return architecture_binary_files_dict
