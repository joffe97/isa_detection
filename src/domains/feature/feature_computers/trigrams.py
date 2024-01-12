import numpy as np

from .feature_computer import FeatureComputer
from helpers import pickle_func


class Trigrams(FeatureComputer):
    @pickle_func
    def compute(binary_file: str) -> dict[str, float]:
        trigram_counts = np.zeros(int("0xffffff", 16) + 1, dtype=np.uint32)
        trigram_count = 0
        with open(binary_file, "rb") as f:
            previous_bytes = [None, None]
            all_previous_bytes_set = False
            while current_byte := f.read(1):
                if all_previous_bytes_set or (all_previous_bytes_set := all(previous_byte is not None for previous_byte in previous_bytes)):
                    trigram_count += 1
                    trigram_int = (
                        ord(previous_bytes[1]) << 16) + (ord(previous_bytes[0]) << 8) + ord(current_byte)
                    trigram_counts[trigram_int] += 1
                previous_bytes[1] = previous_bytes[0]
                previous_bytes[0] = current_byte
        trigrams_f64 = trigram_counts.astype(np.float64)
        trigrams_f64 /= trigram_count
        return dict((f"trigram_0x{hex(i)[2:].zfill(6)}", trigram) for i, trigram in enumerate(trigrams_f64))
