import statistics
import math
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
        return statistics.fmean(result.precision() for result in self.collection)

    def variance(self) -> float:
        mean_precison = self.mean_precision()
        return sum((result.precision() - mean_precison) ** 2 for result in self.collection) / len(
            self.collection
        )

    def standard_deviation(self) -> float:
        return math.sqrt(self.variance())

    def get_outputs_set(self) -> set[object]:
        return set().union(*(result.get_outputs_set() for result in self.collection))

    def print_results(self):
        for result in self.collection:
            result.print()
        print()

    def base_line(self) -> float:
        return 1 / len(self.get_outputs_set())
