from typing import Any, Callable, Optional
from domains.dataset.classes.architecture_file_datas import ArchitectureFileDatas
from domains.dataset.classes.file_data import FileData
from domains.label.label_entry import LabelEntry


class ArchitectureFileDatasMapping(dict[str, ArchitectureFileDatas]):
    @classmethod
    def from_architecture_file_datas_list(
        cls, architecture_file_datas_list: list[ArchitectureFileDatas]
    ) -> "ArchitectureFileDatasMapping":
        architecture_file_datas_mapping = dict(
            (architecture_file_datas.architecture, architecture_file_datas)
            for architecture_file_datas in architecture_file_datas_list
        )
        return cls(architecture_file_datas_mapping)

    def get_means_grouped_by_label_entry(
        self,
        label_entry: LabelEntry,
        label_entry_group_func: Callable[[Any], Optional[Any]] = lambda label_value: label_value,
        label_mapping: Optional[dict[Any, str]] = None,
    ) -> dict[str, list[float]]:
        if label_mapping is None:
            label_mapping = dict()

        label_means_tuples = [
            (architecture_file_datas.find_label_value(label_entry), architecture_file_datas.mean_data())
            for architecture_file_datas in self.values()
        ]

        label_means_mapping: dict[str, ArchitectureFileDatas] = dict()
        for label_value, mean_data in label_means_tuples:
            if label_value is None:
                continue

            label_entry_group = label_entry_group_func(label_value)
            if label_entry_group is None:
                continue
            label_entry_group = label_mapping.get(label_entry_group, label_entry_group)

            label_means_mapping.setdefault(label_entry_group, ArchitectureFileDatas(label_entry_group, []))
            label_means_mapping[label_entry_group].append_file_data(FileData.with_none_path(mean_data))

        return dict(
            (label_value, architecture_file_datas.mean_data())
            for label_value, architecture_file_datas in label_means_mapping.items()
        )

    def mean_datas(self) -> dict[str, list[float]]:
        return dict(
            (architecture, architecture_file_datas.mean_data())
            for architecture, architecture_file_datas in self.items()
        )

    def file_data_mappings(self) -> dict[str, list[tuple[str, list[float]]]]:
        return dict(
            (architecture, architecture_file_datas.file_data_mappings())
            for architecture, architecture_file_datas in self.items()
        )
