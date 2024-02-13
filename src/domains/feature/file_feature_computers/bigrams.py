import numpy as np

from domains.caching import cache_func
from domains.feature.feature_entry import FeatureEntry
from .file_feature_computer import FileFeatureComputer


class Bigrams(FileFeatureComputer):
    @staticmethod
    @cache_func()
    def compute(binary_file: str) -> dict[str, FeatureEntry]:
        bigram_counts = np.zeros(int("0xffff", 16) + 1, dtype=np.uint64)
        bigram_count = 0
        with open(binary_file, "rb") as f:
            previous_byte = None
            while current_byte := f.read(1):
                if previous_byte is not None:
                    bigram_count += 1
                    bigram_int = (ord(previous_byte) << 8) + ord(current_byte)
                    bigram_counts[bigram_int] += 1
                previous_byte = current_byte
        bigrams_f64 = bigram_counts.astype(np.float64)
        bigrams_f64 /= bigram_count
        return dict((f"bigram_0x{(hex(i)[2:]).zfill(4)}", FeatureEntry(bigram, i)) for i, bigram in enumerate(bigrams_f64))
