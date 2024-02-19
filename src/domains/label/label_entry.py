from enum import Enum
from functools import cache


class LabelEntry(Enum):
    WORD_SIZE = "wordsize"
    ENDIANNESS = "endianness"
    INSTRUCTION_SIZE = "instruction_size"
    IS_VARIABLE_INSTRUCTION_SIZE = "is_variable_instruction_size"
    VARIABLE_INSTRUCTION_SIZE = "variable_instruction_size"
    ARCHITECTURE_TEXT = "architecture_text"
    ARCHITECTURE_ID = "architecture"

    @classmethod
    def __get_exclusive_groups(cls) -> set[tuple["LabelEntry", ...]]:
        return {(cls.INSTRUCTION_SIZE, cls.VARIABLE_INSTRUCTION_SIZE)}

    @classmethod
    def __get_dependency_groups(cls) -> set[tuple["LabelEntry", ...]]:
        return {
            (cls.INSTRUCTION_SIZE, cls.IS_VARIABLE_INSTRUCTION_SIZE),
            (cls.VARIABLE_INSTRUCTION_SIZE, cls.IS_VARIABLE_INSTRUCTION_SIZE),
        }

    @classmethod
    def __get_label_groups_map(
        cls, groups: set[tuple["LabelEntry", ...]]
    ) -> dict["LabelEntry", set["LabelEntry"]]:
        label_groups_map = dict((label_entry, {label_entry}) for label_entry in cls)
        for group in groups:
            for group_entry in group:
                label_groups_map[group_entry].update(group)
        return label_groups_map

    @classmethod
    @cache
    def __get_label_exclusive_groups_map(cls) -> dict["LabelEntry", set["LabelEntry"]]:
        return cls.__get_label_groups_map(cls.__get_exclusive_groups())

    @classmethod
    @cache
    def __get_label_dependency_groups_map(cls) -> dict["LabelEntry", set["LabelEntry"]]:
        return cls.__get_label_groups_map(cls.__get_dependency_groups())

    def get_exclusive_entries(self) -> set["LabelEntry"]:
        return self.__get_label_exclusive_groups_map().get(self, {self})

    def get_dependency_entries(self) -> set["LabelEntry"]:
        return self.__get_label_dependency_groups_map().get(self, {self})

    @classmethod
    @cache
    def __get_value_to_member_map(cls) -> dict[str, "LabelEntry"]:
        return {member.value: member for member in cls}

    @classmethod
    def from_str(cls, string: str) -> "LabelEntry":
        return cls.__get_value_to_member_map()[string]

    def __str__(self) -> str:
        return self.value

    @classmethod
    @cache
    def all(cls) -> set["LabelEntry"]:
        return set(label for label in cls)

    @classmethod
    @cache
    def all_names(cls) -> set[str]:
        return set(map(str, cls))
