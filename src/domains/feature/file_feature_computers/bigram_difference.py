from domains.caching import cache_func
from domains.feature.feature_entry import FeatureEntry
from helpers.prime import FIRST_PRIMES, prime_factors
from .file_feature_computer import FileFeatureComputer


class BigramDifference(FileFeatureComputer):
    @staticmethod
    def create_bigram_difference_lists(binary_file: str) -> list[dict[int, int]]:
        bigram_difference_lists: list[dict[int, int]] = [dict() for _ in range(256**2)]
        prev_bigram_indexes = [-1 for _ in range(256**2)]
        current_bigram_index = -1

        with open(binary_file, "rb") as f:
            prev_byte = None
            while current_byte := f.read(1):
                current_byte_int = ord(current_byte)
                if prev_byte is None:
                    prev_byte = current_byte
                    continue

                prev_byte_int = ord(prev_byte)
                prev_byte = current_byte
                current_bigram_int = (prev_byte_int << 8) | current_byte_int

                current_bigram_index += 1
                prev_bigram_index = prev_bigram_indexes[current_bigram_int]

                if prev_bigram_index != -1:
                    bigram_difference = current_bigram_index - prev_bigram_index
                    bigram_difference_lists[current_bigram_int].setdefault(bigram_difference, 0)
                    bigram_difference_lists[current_bigram_int][bigram_difference] += 1

                prev_bigram_indexes[current_bigram_int] = current_bigram_index
        return bigram_difference_lists

    @staticmethod
    def create_bigram_difference_frequency_lists(
        bigram_difference_lists: list[dict[int, int]]
    ) -> list[dict[int, float]]:
        bigram_difference_frequency_lists = [dict() for _ in range(256**2)]
        for current_bigram, bigram_difference in enumerate(bigram_difference_lists):
            total_bigram_difference_count = sum(bigram_difference.values())
            for bigram_difference, bigram_difference_count in bigram_difference.items():
                bigram_difference_frequency = bigram_difference_count / total_bigram_difference_count
                bigram_difference_frequency_lists[current_bigram][
                    bigram_difference
                ] = bigram_difference_frequency
        return bigram_difference_frequency_lists

    @staticmethod
    def create_bigram_difference_factorized_dict(
        bigram_difference_frequency_lists: list[dict[int, float]]
    ) -> dict[str, FeatureEntry]:
        bigram_difference_factorized_dict = dict()
        for current_bigram, bigram_difference_frequencies in enumerate(bigram_difference_frequency_lists):
            for prime_index, prime in enumerate(FIRST_PRIMES):
                dict_key = f"bigramdiff_{current_bigram}_{prime}"
                bigram_difference_factorized_dict.setdefault(
                    dict_key, FeatureEntry(0, prime_index + len(FIRST_PRIMES) * current_bigram)
                )
            for bigram_difference, bigram_difference_frequency in bigram_difference_frequencies.items():
                prime_factors_set = prime_factors(bigram_difference, max(FIRST_PRIMES))
                for prime in FIRST_PRIMES:
                    dict_key = f"bigramdiff_{current_bigram}_{prime}"
                    if prime in prime_factors_set:
                        bigram_difference_factorized_dict[dict_key].value += bigram_difference_frequency
        return bigram_difference_factorized_dict

    @cache_func()
    def compute(self, binary_file: str) -> dict[str, FeatureEntry]:
        bigram_difference_lists = self.create_bigram_difference_lists(binary_file)
        bigram_difference_frequency_lists = self.create_bigram_difference_frequency_lists(
            bigram_difference_lists
        )
        return self.create_bigram_difference_factorized_dict(bigram_difference_frequency_lists)
