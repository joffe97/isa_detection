import numpy as np

from domains.caching import cache_func
from domains.feature.feature_entry import FeatureEntry
from .file_feature_computer import FileFeatureComputer


class ByteFrequencyDistribution(FileFeatureComputer):
    @staticmethod
    @cache_func()
    def compute(binary_file: str) -> dict[str, FeatureEntry]:
        byte_frequency_distribution_u64 = np.zeros(256, dtype=np.uint64)
        with open(binary_file, "rb") as f:
            while current_byte := f.read(1):
                current_byte_int = ord(current_byte)
                byte_frequency_distribution_u64[current_byte_int] += 1
        byte_frequency_distribution_f64 = byte_frequency_distribution_u64.astype(
            np.float64)
        byte_frequency_distribution_f64 /= sum(byte_frequency_distribution_f64)
        return dict((f"bfd_{i}", FeatureEntry(value, i)) for i, value in enumerate(byte_frequency_distribution_f64))
