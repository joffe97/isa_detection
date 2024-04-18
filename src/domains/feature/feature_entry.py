class FeatureEntry:
    def __init__(self, value: float, numerical_identifier: int) -> None:
        self.value = value
        self.numerical_identifier = numerical_identifier

    @classmethod
    def with_irrelevant_numerical_identifier(cls, value: float) -> "FeatureEntry":
        return cls(value, -1)
