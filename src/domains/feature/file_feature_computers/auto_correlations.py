from domains.caching.cache_func_decorator import cache_func
from domains.feature.feature_entry import FeatureEntry
from domains.feature.file_feature_computers.auto_correlation import AutoCorrelation
from domains.feature.file_feature_computers.file_feature_computer import FileFeatureComputer


class AutoCorrelations(FileFeatureComputer):
    def __init__(self, lags: list[int]) -> None:
        super().__init__()
        self.lags = lags

    @cache_func()
    def compute(self, binary_file: str) -> dict[str, FeatureEntry]:
        with open(binary_file, "rb") as f:
            data = f.read()

        return dict(
            (
                f"autocorr_{lag}",
                FeatureEntry.with_irrelevant_numerical_identifier(
                    AutoCorrelation(lag).compute_for_bytes(data)
                ),
            )
            for lag in self.lags
        )
