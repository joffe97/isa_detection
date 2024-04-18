from domains.caching import cache_func
from domains.feature.feature_entry import FeatureEntry
from . import FileFeatureComputer, Trigrams


class TrigramsNonZero(FileFeatureComputer):
    @cache_func()
    def compute(self, binary_file: str) -> dict[str, FeatureEntry]:
        trigrams = Trigrams.compute_without_cache(binary_file)
        return dict(
            (f"trigram_non_zero{key.lstrip('trigram')}", feature_entry)
            for key, feature_entry in trigrams.items()
            if feature_entry.value != 0
        )
