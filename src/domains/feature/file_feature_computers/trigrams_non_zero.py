from domains.caching import cache_func
from . import FileFeatureComputer, Trigrams


class TrigramsNonZero(FileFeatureComputer):
    @staticmethod
    @cache_func()
    def compute(binary_file: str) -> dict[str, float]:
        trigrams = Trigrams.compute_without_cache(binary_file)
        return dict((f"trigram_non_zero{key.lstrip('trigram')}", value) for key, value in trigrams.items() if value != 0)
