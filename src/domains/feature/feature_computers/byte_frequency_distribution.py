import numpy as np

from .feature_computer import FeatureComputer
from helpers import pickle_func


class ByteFrequencyDistribution(FeatureComputer):
    @pickle_func
    def compute(binary_file: str) -> dict[str, float]:
        byte_frequency_distribution_u64 = np.zeros(256, dtype=np.uint64)
        with open(binary_file, "rb") as f:
            while current_byte := f.read(1):
                current_byte_int = ord(current_byte)
                byte_frequency_distribution_u64[current_byte_int] += 1
        byte_frequency_distribution_f64 = byte_frequency_distribution_u64.astype(
            np.float64)
        byte_frequency_distribution_f64 /= sum(byte_frequency_distribution_f64)
        return dict((str(i), entry) for i, entry in enumerate(byte_frequency_distribution_f64))
