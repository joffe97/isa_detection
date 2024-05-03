import csv
from functools import cache
from random import randint
from typing import Optional
from config import Config
from domains.label.architecture_labels import ArchitectureLabels
from domains.label.label_entry import LabelEntry
from domains.label.label_loaders.label_loader import LabelLoader


class CorpusLabels(LabelLoader):
    def __init__(self, included_labels: Optional[set[LabelEntry]] = None) -> None:
        super().__init__()
        self.included_labels = included_labels

        self.raise_exception_if_invalid_included_labels()

    @classmethod
    def with_default_included_labels(
        cls, included_labels: Optional[set[LabelEntry]] = None
    ) -> "CorpusLabels":
        if included_labels is None:
            included_labels = set()
        default_included_labels = {LabelEntry.ARCHITECTURE_TEXT}
        return cls(default_included_labels.union(included_labels))

    def raise_exception_if_invalid_included_labels(self) -> None:
        if self.included_labels is None:
            return
        invalid_label_iters = (
            label.get_exclusive_entries().difference([label]) for label in self.included_labels
        )
        invalid_labels = set(b for a in invalid_label_iters for b in a)
        if len(invalid_labels.intersection(invalid_label_iters)) != 0:
            raise ValueError(
                f"included_labels contains one or more pairs of labels that cannot be together: {self.included_labels}"
            )

    @property
    def included_labels_strs(self) -> Optional[set[str]]:
        if self.included_labels is None:
            return None
        return set(included_label.value for included_label in self.included_labels)

    def is_included_label_str(self, label_str: str) -> bool:
        return (
            included_labels_strs := self.included_labels_strs
        ) is None or label_str in included_labels_strs

    @cache  # pylint: disable=W1518
    def load(self) -> list[ArchitectureLabels]:
        all_valid_corpus_labels: list[ArchitectureLabels] = []
        with open(Config.CORPUS_CLASSIFICATION_PATH, "r") as csv_file:
            columns_to_read = 6
            csv_reader = csv.reader(csv_file, delimiter=",")

            head_translation_dict = {
                "Common name": LabelEntry.ARCHITECTURE_TEXT.value,
                "Instruction Size": LabelEntry.INSTRUCTION_SIZE.value,
            }
            head = [
                (
                    column_name.lower()
                    if column_name not in head_translation_dict
                    else head_translation_dict[column_name]
                )
                for column_name in next(csv_reader)[:columns_to_read]
            ]

            for line in csv_reader:
                item_strs = line[:columns_to_read]
                architecture_labels = ArchitectureLabels()
                # invalid_items = False
                isa = None
                for i, label_value in enumerate(item_strs):
                    label_name = head[i]

                    if label_name == "isa":
                        isa = label_value
                        continue
                    # elif not self.is_included_label_str(label_name):
                    #     continue
                    elif label_name == LabelEntry.ENDIANNESS.value:
                        if label_value == "LE":
                            label_value = "Little"
                        elif label_value == "BE":
                            label_value = "Big"
                        else:
                            # invalid_items = True
                            # break
                            continue
                    elif label_name == LabelEntry.INSTRUCTION_SIZE.value:
                        instruction_width_strs = label_value.split("-")
                        try:
                            instruction_widths = tuple(map(int, instruction_width_strs))
                            if len(instruction_widths) == 1:
                                architecture_labels[LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE] = False
                                architecture_labels[LabelEntry.FIXED_INSTRUCTION_SIZE] = instruction_widths[0]
                            elif len(instruction_widths) == 2:
                                architecture_labels[LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE] = True
                                architecture_labels[LabelEntry.VARIABLE_INSTRUCTION_SIZE] = instruction_widths
                            else:
                                raise ValueError(f"Instruction width cannot be parsed: {label_value}")
                        except ValueError:
                            # invalid_items = True
                            # break
                            continue
                    elif label_name == LabelEntry.WORD_SIZE.value:
                        try:
                            label_value = int(label_value)
                        except ValueError:
                            # invalid_items = True
                            # break
                            continue
                    elif label_name == LabelEntry.ARCHITECTURE_TEXT.value:
                        if label_value == "":
                            isa = next(
                                item_tmp for i_tmp, item_tmp in enumerate(item_strs) if head[i_tmp] == "isa"
                            )
                            label_value = isa if isa else f"unknown_{str(randint(1, 999_999)).zfill(6)}"
                    elif label_name not in LabelEntry.all_names():
                        continue
                    architecture_labels[LabelEntry.from_str(label_name)] = label_value
                # if not invalid_items:
                all_valid_corpus_labels.append(architecture_labels)
                if isa and (isa != architecture_labels.get(LabelEntry.ARCHITECTURE_TEXT)):
                    architecture_labels_copy = architecture_labels.copy()
                    architecture_labels_copy[LabelEntry.ARCHITECTURE_TEXT] = isa
                    all_valid_corpus_labels.append(architecture_labels_copy)

        if self.included_labels is not None:
            corpus_labels = []
            for label_dict in all_valid_corpus_labels:
                labels_to_keep = dict(
                    filter(
                        lambda key_value: key_value[0].value in self.included_labels_strs,
                        label_dict.items(),
                    )
                )
                if len(labels_to_keep) != len(self.included_labels):
                    continue
                corpus_labels.append(ArchitectureLabels(labels_to_keep))
        else:
            corpus_labels = all_valid_corpus_labels

        return corpus_labels
