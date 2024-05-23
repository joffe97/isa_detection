from typing import Union


class FeatureEntry:
    def __init__(self, value: Union[float, int], numerical_identifier: int) -> None:
        self.value = value
        self.numerical_identifier = numerical_identifier

    @classmethod
    def with_irrelevant_numerical_identifier(cls, value: Union[float, int]) -> "FeatureEntry":
        return cls(value, -1)
