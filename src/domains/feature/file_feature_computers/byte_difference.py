from domains.caching import cache_func
from domains.feature.feature_entry import FeatureEntry
from helpers.prime import FIRST_PRIMES, prime_factors
from .file_feature_computer import FileFeatureComputer


class ByteDifference(FileFeatureComputer):
    @staticmethod
    def create_byte_difference_lists(binary_file: str) -> list[dict[int, int]]:
        byte_difference_lists: list[dict[int, int]] = [dict() for _ in range(256)]
        prev_byte_indexes = [-1 for _ in range(256)]
        current_byte_index = -1

        with open(binary_file, "rb") as f:
            while current_byte := f.read(1):
                current_byte_int = ord(current_byte)
                current_byte_index += 1
                prev_byte_index = prev_byte_indexes[current_byte_int]

                if prev_byte_index != -1:
                    byte_difference = current_byte_index - prev_byte_index
                    byte_difference_lists[current_byte_int].setdefault(byte_difference, 0)
                    byte_difference_lists[current_byte_int][byte_difference] += 1

                prev_byte_indexes[current_byte_int] = current_byte_index
        return byte_difference_lists

    @staticmethod
    def create_byte_difference_frequency_lists(
        byte_difference_lists: list[dict[int, int]]
    ) -> list[dict[int, float]]:
        byte_difference_frequency_lists = [dict() for _ in range(256)]
        for current_byte, byte_difference in enumerate(byte_difference_lists):
            total_byte_difference_count = sum(byte_difference.values())
            for byte_difference, byte_difference_count in byte_difference.items():
                byte_difference_frequency = byte_difference_count / total_byte_difference_count
                byte_difference_frequency_lists[current_byte][byte_difference] = byte_difference_frequency
        return byte_difference_frequency_lists

    @staticmethod
    def create_byte_difference_factorized_dict(
        byte_difference_frequency_lists: list[dict[int, float]]
    ) -> dict[str, FeatureEntry]:
        byte_difference_factorized_dict = dict()
        for current_byte, byte_difference_frequencies in enumerate(byte_difference_frequency_lists):
            for j, prime in enumerate(FIRST_PRIMES):
                dict_key = f"bytediff_{current_byte}_{prime}"
                byte_difference_factorized_dict.setdefault(
                    dict_key, FeatureEntry(0, j + len(FIRST_PRIMES) * current_byte)
                )
            for byte_difference, byte_difference_frequency in byte_difference_frequencies.items():
                prime_factors_set = prime_factors(byte_difference, max(FIRST_PRIMES))
                for prime in FIRST_PRIMES:
                    dict_key = f"bytediff_{current_byte}_{prime}"
                    if prime in prime_factors_set:
                        byte_difference_factorized_dict[dict_key].value += byte_difference_frequency
        return byte_difference_factorized_dict

    @cache_func()
    def compute(self, binary_file: str) -> dict[str, FeatureEntry]:
        byte_difference_lists = self.create_byte_difference_lists(binary_file)
        byte_difference_frequency_lists = self.create_byte_difference_frequency_lists(byte_difference_lists)
        return self.create_byte_difference_factorized_dict(byte_difference_frequency_lists)
