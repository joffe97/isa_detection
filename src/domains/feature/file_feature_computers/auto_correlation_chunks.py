from pandas import Series
from domains.caching.cache_func_decorator import cache_func
from domains.feature.feature_entry import FeatureEntry
from domains.feature.file_feature_computers.file_feature_computer import FileFeatureComputer


class AutoCorrelationChunks(FileFeatureComputer):
    def __init__(self, chunk_size: int, chunk_count: int, lag: int) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_count = chunk_count
        self.lag = lag

    @cache_func(use_class_identifier_method=True)
    def compute(self, binary_file: str) -> dict[str, FeatureEntry]:
        NAN_REPLACER = -2
        chunk_auto_correlations = dict(
            (f"autocorr_{chunk_i}", FeatureEntry(NAN_REPLACER, chunk_i))
            for chunk_i in range(self.chunk_count)
        )

        with open(binary_file, "rb") as f:
            chunk_i = 0
            while chunk_i < self.chunk_count and (data := f.read(self.chunk_size)):
                data_ints = list(iter(data))
                series = Series(data_ints)
                auto_correlation = series.autocorr(self.lag)

                chunk_auto_correlations[f"autocorr_{chunk_i}"] = FeatureEntry(auto_correlation, chunk_i)
                chunk_i += 1

        return chunk_auto_correlations
