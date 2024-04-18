from abc import ABC, abstractmethod

from domains.dataset.binary_file_dataset import BinaryFileDataset


class AutoCorrelationCreatorBase(ABC):
    @abstractmethod
    def create(self, dataset: BinaryFileDataset):
        pass
