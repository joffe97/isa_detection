import statistics
from typing import Any, Callable
from domains.model.info.isa_model_result import ISAModelResult


class ISAModelResultCollection:
    def __init__(self, collection: list[ISAModelResult]) -> None:
        self.collection = collection

    def get_collection_sorted_by(
        self, key: Callable[[ISAModelResult], Any] = lambda result: result.architecture_id
    ) -> list[ISAModelResult]:
        collection_copy = self.collection.copy()
        collection_copy.sort(key=key)
        return collection_copy

    def mean_precision(self) -> float:
        return statistics.fmean(result.precision for result in self.collection)

    def print_results(self):
        for result in self.collection:
            result.print()
        print()
