from abc import ABC, abstractmethod
from pathlib import Path
import itertools

from config import Config
from domains.dataset.binary_file_dataset import BinaryFileDataset
from .researcher import Researcher


class FileResearcher(Researcher, ABC):
    @abstractmethod
    def _research_file(self, architecture: str, path: Path, group_name: str):
        pass

    def _research_files_for_architecture(self, architecture: str, paths: list[Path], group_name: str):
        return list(
            map(self._research_file, itertools.repeat(architecture), paths, itertools.repeat(group_name))
        )

    def research(self, dataset: BinaryFileDataset):
        for architecture, path_strs in dataset.create_architecture_paths_mapping().items():
            paths = list(map(Path, path_strs))
            self._research_files_for_architecture(architecture, paths, dataset.identifier())
