from abc import ABC, abstractmethod

from domains.label.architecture_labels import ArchitectureLabels
from domains.label.label_entry import LabelEntry


class LabelLoader(ABC):
    @abstractmethod
    def load(self) -> list[ArchitectureLabels]:
        pass

    def load_as_architecture_labels_mapping(self) -> dict[str, ArchitectureLabels]:
        architecture_labels = self.load()
        return dict((label.architecture_text, label) for label in architecture_labels)

    def load_as_binary_file_labels_mapping(
        self, architecture_binary_files_mapping: dict[str, list[str]]
    ) -> list[tuple[str, ArchitectureLabels]]:
        architecture_labels_mapping = self.load_as_architecture_labels_mapping()

        architecture_binary_files_dict_items = sorted(
            architecture_binary_files_mapping.items(), key=lambda item: item[0]
        )

        binary_file_labels_list: list[tuple[str, ArchitectureLabels]] = []
        included_labels = set()
        architecture_count = 0
        for architecture_text, binary_files in architecture_binary_files_dict_items:
            labels = architecture_labels_mapping.get(architecture_text)
            if labels is None:
                continue
            labels_copy = labels.copy()

            architecture_count += 1
            labels_copy[LabelEntry.ARCHITECTURE_ID] = architecture_count

            if len(included_labels) == 0:
                included_labels = labels_copy.included_labels()

            for include_label in list(included_labels):
                if include_label not in labels_copy:
                    included_labels.remove(include_label)
                    for _, other_labels in binary_file_labels_list:
                        other_labels.pop(include_label, None)
                for label in list(labels_copy.keys()):
                    if label not in included_labels:
                        labels_copy.pop(label, None)

            for binary_file in binary_files:
                binary_file_labels_list.append((binary_file, labels_copy))

        return binary_file_labels_list
