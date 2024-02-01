class ISAModelResult:
    def __init__(self, architecture_id: int, precision: float) -> None:
        self.architecture_id = architecture_id
        self.precision = precision

    def print(self):
        print(f"{self.architecture_id}\t{self.precision}")
