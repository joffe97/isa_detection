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

    def get_all_correct_outputs(self) -> list[object]:
        all_correct_outputs = []
        for result in self.collection:
            all_correct_outputs.extend(result.correct_outputs)
        return all_correct_outputs

    def get_most_common_output(self) -> object:
        all_correct_outputs = self.get_all_correct_outputs()
        return max(set(all_correct_outputs), key=all_correct_outputs.count)

    def _get_most_common_output_count(self) -> int:
        all_correct_outputs = self.get_all_correct_outputs()
        most_common_output = self.get_most_common_output()
        return len(list(filter(lambda x: x == most_common_output, all_correct_outputs)))

    def base_line(self) -> float:
        all_random_base_line = 1 / len(self.get_outputs_set())
        all_most_common_base_line = self._get_most_common_output_count() / len(self.get_all_correct_outputs())
        return max(all_random_base_line, all_most_common_base_line)
