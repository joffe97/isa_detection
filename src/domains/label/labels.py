from typing import Callable
import csv
from random import randint

from config import Config


class Labels:
    __labels_cache = dict()

    @classmethod
    def __get_label_or_create_if_not_exist(cls, label_creation_method: Callable[[], list[dict[str, object]]]) -> list[dict[str, object]]:
        label_id = label_creation_method.__name__
        label_value = Labels.__labels_cache.get(label_id)
        if label_value is None:
            label_value = label_creation_method()
            Labels.__labels_cache[label_id] = label_value
            a = Labels.__labels_cache
        return label_value

    @classmethod
    def get_corpus_labels(cls) -> list[dict[str, object]]:
        return cls.__get_label_or_create_if_not_exist(cls.__create_corpus_labels)

    @classmethod
    def get_isa_detect_csv_labels(cls) -> list[dict[str, object]]:
        return cls.__get_label_or_create_if_not_exist(cls.__create_isa_detect_csv_labels)

    @classmethod
    def get_labels_combined(cls) -> list[dict[str, object]]:
        return cls.__get_label_or_create_if_not_exist(cls.__create_labels_combined)

    @classmethod
    def get_corpus_exclusive(cls) -> list[dict[str, object]]:
        corpus_labels = cls.get_corpus_labels()
        isa_detect_labels = cls.get_isa_detect_csv_labels()
        isa_detect_architecture_texts = {
            *[isa_detect_label["architecture_text"] for isa_detect_label in isa_detect_labels], "s-390"}
        return [label for label in corpus_labels if label["architecture_text"] not in isa_detect_architecture_texts]

    @classmethod
    def __create_corpus_labels(cls) -> list[dict[str, object]]:
        corpus_labels = []
        with open(Config.CORPUS_CLASSIFICATION_PATH) as csv_file:
            columns_to_read = 6
            head_translation_dict = {"Common name": "architecture_text",
                                     "Instruction Size": "instruction_size"}

            csv_reader = csv.reader(csv_file, delimiter=",")
            head = [(column_name.lower() if column_name not in head_translation_dict else head_translation_dict[column_name])
                    for column_name in next(csv_reader)[:columns_to_read]]
            for line in csv_reader:
                items: list = line[:columns_to_read]
                for i, item in enumerate(items):
                    column_name = head[i]
                    if (item.isnumeric()):
                        item = int(item)
                    if (column_name == "endianness"):
                        if item == "LE":
                            item = "Little"
                        elif item == "BE":
                            item = "Big"
                        else:
                            items = None
                            break
                    if (column_name in ["wordsize", "instruction_size"]):
                        if not isinstance(item, int):
                            items = None
                            break
                    if (column_name == "architecture_text"):
                        if item == "":
                            isa = next(item_tmp for i_tmp, item_tmp in enumerate(
                                items) if head[i_tmp] == "isa")
                            item = isa if isa else f"unknown_{str(randint(1, 999_999)).zfill(6)}"
                    items[i] = item
                if items is not None:
                    corpus_labels.append(dict(zip(head, items)))
        return corpus_labels

    @classmethod
    def __create_isa_detect_csv_labels(cls) -> list[dict[str, object]]:
        return [
            {"architecture_text": "alpha", "wordsize": 64, "endianness":  "Little"},
            {"architecture_text": "amd64", "wordsize": 64, "endianness":  "Little"},
            {"architecture_text": "arm64", "wordsize": 64, "endianness":  "Little"},
            {"architecture_text": "armel", "wordsize": 32, "endianness":  "Little"},
            {"architecture_text": "armhf", "wordsize": 32, "endianness":  "Little"},
            {"architecture_text": "hppa", "wordsize": 32, "endianness":  "Big"},
            {"architecture_text": "i386", "wordsize": 32, "endianness":  "Little"},
            {"architecture_text": "ia64", "wordsize": 64, "endianness":  "Little"},
            {"architecture_text": "m68k", "wordsize": 32, "endianness":  "Big"},
            {"architecture_text": "mips", "wordsize": 32, "endianness":  "Big"},
            {"architecture_text": "mips64el",
                "wordsize": 64, "endianness":  "Little"},
            {"architecture_text": "mipsel", "wordsize": 32, "endianness":  "Little"},
            {"architecture_text": "powerpc", "wordsize": 32, "endianness":  "Big"},
            {"architecture_text": "powerpcspe",
                "wordsize": 32, "endianness":  "Big"},
            {"architecture_text": "ppc64", "wordsize": 64, "endianness":  "Big"},
            {"architecture_text": "ppc64el", "wordsize": 64, "endianness":  "Little"},
            {"architecture_text": "riscv64", "wordsize": 64, "endianness":  "Little"},
            {"architecture_text": "s390", "wordsize": 32, "endianness":  "Big"},
            {"architecture_text": "s390x", "wordsize": 64, "endianness":  "Big"},
            {"architecture_text": "sh4", "wordsize": 32, "endianness":  "Little"},
            {"architecture_text": "sparc", "wordsize": 32, "endianness":  "Big"},
            {"architecture_text": "sparc64", "wordsize": 64, "endianness":  "Big"},
            {"architecture_text": "x32", "wordsize": 32, "endianness":  "Little"},
        ]

    @classmethod
    def __create_labels_combined(cls) -> list[dict[str, object]]:
        def labels_list_to_dict(labels_list: list[dict]) -> dict[str, dict]:
            return dict((labels["architecture_text"], labels) for labels in labels_list)

        labels_combined_dict = dict()
        for include_labels in [cls.get_isa_detect_csv_labels(), cls.get_corpus_labels()]:
            labels_combined_dict.update(labels_list_to_dict(include_labels))

        labels_combined = labels_combined_dict.values()

        return list(labels_combined)
