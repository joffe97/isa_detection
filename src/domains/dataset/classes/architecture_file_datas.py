import statistics
from typing import Any, Optional
from domains.dataset.classes.file_data import FileData
from domains.label.label_entry import LabelEntry
from domains.label.label_loaders.corpus_labels import CorpusLabels


class ArchitectureFileDatas:
    def __init__(self, architecture: str, file_datas: list[FileData]) -> None:
        self.architecture = architecture
        self.file_datas = file_datas

    def find_label_value(self, label_entry: LabelEntry) -> Optional[Any]:
        labels = CorpusLabels.with_default_included_labels(
            {label_entry}
        ).load_as_architecture_labels_mapping()
        architecture_labels = labels.get(self.architecture)
        return architecture_labels and architecture_labels.get(label_entry)

    def append_file_data(self, file_data: FileData) -> None:
        self.file_datas.append(file_data)

    def get_max_file_datas_len(self) -> int:
        return max(len(file_data.data) for file_data in self.file_datas)

    def mean_data(self) -> list[float]:
        return [
            statistics.fmean(file_data.data[index] for file_data in self.file_datas)
            for index in range(self.get_max_file_datas_len())
        ]

    def file_data_mappings(self) -> list[tuple[str, list[float]]]:
        return list(map(FileData.file_data_mapping, self.file_datas))
