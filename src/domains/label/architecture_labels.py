from typing import Optional
from domains.label.label_entry import LabelEntry


class ArchitectureLabels(dict[LabelEntry, object]):
    def __init__(self, labels: Optional[dict[LabelEntry, object]] = None) -> None:
        if labels is None:
            labels = dict()
        super().__init__(labels)

    @property
    def architecture_text(self) -> str:
        return self[LabelEntry.ARCHITECTURE_TEXT]  # type: ignore

    @architecture_text.setter
    def architecture_text(self, value: str) -> None:
        self[LabelEntry.ARCHITECTURE_TEXT] = value

    def included_labels(self) -> set[LabelEntry]:
        return set(self.keys())

    def copy(self) -> "ArchitectureLabels":
        return ArchitectureLabels(super().copy())
