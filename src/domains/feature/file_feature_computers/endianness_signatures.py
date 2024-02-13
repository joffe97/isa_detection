import numpy as np

from domains.feature.feature_entry import FeatureEntry

from .file_feature_computer import FileFeatureComputer
from domains.caching import cache_func


class EndiannessSignatures(FileFeatureComputer):
    @staticmethod
    @cache_func()
    def compute(binary_file: str) -> dict[str, FeatureEntry]:
        endianness_signature_names = [
            # "be_one", "be_stack", "le_one", "le_stack"]
            "bigram_0x0001", "bigram_0xfffe", "bigram_0x0100", "bigram_0xfeff"]
        endianness_signature_values = [int(hex_str, 16) for hex_str in [
            "0x0001", "0xfffe", "0x0100", "0xfeff"]]
        bigram_counts = np.zeros(4, dtype=np.uint64)
        bigram_count = 0
        with open(binary_file, "rb") as f:
            previous_byte = None
            while current_byte := f.read(1):
                if previous_byte is not None:
                    bigram_count += 1
                    bigram_int = (ord(previous_byte) << 8) + ord(current_byte)
                    for i in range(4):
                        if endianness_signature_values[i] == bigram_int:
                            bigram_counts[i] += 1
                            break
                previous_byte = current_byte
        endianness_signatures_f64 = bigram_counts.astype(np.float64)
        endianness_signatures_f64 /= bigram_count
        return dict((endianness_signature_names[i], FeatureEntry(endianness_signatures_f64[i], numeric_identifier)) for i, numeric_identifier in enumerate(endianness_signature_values))
