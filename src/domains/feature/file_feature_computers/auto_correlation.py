from pandas import Series
from domains.feature.feature_entry import FeatureEntry
from domains.feature.file_feature_computers.file_feature_computer import FileFeatureComputer


class AutoCorrelation(FileFeatureComputer):
    def __init__(self, lag: int = 1) -> None:
        super().__init__()
        self.lag = lag

    def compute_for_bytes(self, data: bytes) -> float:
        data_ints = list(iter(data))
        series = Series(data_ints)
        return series.autocorr(self.lag)

    def compute(self, binary_file: str) -> dict[str, FeatureEntry]:
        with open(binary_file, "rb") as f:
            data = f.read()
        auto_correlation = self.compute_for_bytes(data)
        return {"autocorr": FeatureEntry.with_irrelevant_numerical_identifier(auto_correlation)}
