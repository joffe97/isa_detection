from pathlib import Path
import statistics
from typing import Iterator
from pandas import Series

from domains.dataset.binary_file_dataset import BinaryFileDataset
from domains.dataset.custom.custom_dataset import CustomDataset
from domains.label.labels import Labels
from . import Researcher


class AutoCorrelationMean(Researcher):
    def __init__(self, byte_read_count: int = 512, lag=1) -> None:
        super().__init__()
        self.byte_read_count = byte_read_count
        self.lag = lag

    @staticmethod
    def get_auto_correlation_means(
        auto_correlation_mapping: dict[str, list[float]],
        include_means: bool = True,
        include_architectures_without_instruction_size: bool = True,
    ) -> list[tuple[str, float, str]]:
        architecture_auto_correlation_mean_mapping = dict()
        for architecture, auto_correlations in auto_correlation_mapping.items():
            auto_correlation_mean = statistics.fmean(auto_correlations)
            architecture_auto_correlation_mean_mapping[architecture] = auto_correlation_mean
        architecture_auto_correlation_mean_mapping_tup_sorted = sorted(
            architecture_auto_correlation_mean_mapping.items(), key=lambda x: x[1], reverse=True
        )

        labels = Labels.get_corpus_labels(True)  ##

        data = []
        for (
            architecture,
            auto_correlation_mean,
        ) in architecture_auto_correlation_mean_mapping_tup_sorted:
            instruction_size = next(
                (
                    label.get("instruction_size")
                    for label in labels
                    if label.get("architecture_text") == architecture or label.get("isa") == architecture
                ),
                None,
            )
            if not instruction_size and not include_architectures_without_instruction_size:
                continue

            data.append((architecture, auto_correlation_mean, str(instruction_size)))

        if include_means:
            variable_mean_autocorr_mean = statistics.fmean(
                abs(auto_corr) for _, auto_corr, instruction_size in data if "-" in instruction_size
            )
            fixed_mean_autocorr_mean = statistics.fmean(
                abs(auto_corr) for _, auto_corr, instruction_size in data if "-" not in instruction_size
            )

            instruction_lengths_tuples = [
                (
                    str(instruction_size),
                    statistics.fmean(
                        abs(auto_corr)
                        for _, auto_corr, cur_instruction_size in data
                        if cur_instruction_size == instruction_size
                    ),
                    "size_mean",
                )
                for instruction_size in set(instruction_size_tmp for _, _, instruction_size_tmp in data)
            ]

            data.extend(
                [
                    ("Variable mean", variable_mean_autocorr_mean, "type_mean"),
                    ("Fixed mean", fixed_mean_autocorr_mean, "type_mean"),
                    *instruction_lengths_tuples,
                ]
            )

        return data

    def get_auto_correlation_mapping(self, dataset: BinaryFileDataset) -> dict[str, list[float]]:
        def auto_corr_data(data: bytes) -> float:
            return Series(list(iter(data))).autocorr(self.lag)

        return dataset.create_architecture_func_data_mapping(self.byte_read_count, auto_corr_data)

    def research(self, dataset: BinaryFileDataset):
        data = self.get_auto_correlation_means(
            self.get_auto_correlation_mapping(dataset),
            include_architectures_without_instruction_size=isinstance(dataset, CustomDataset),
        )

        group_name = "_".join(map(str, [self.byte_read_count, self.lag]))
        file_path = self._create_result_path(group_name, dataset.identifier(), ".txt")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            for architecture, auto_correlation_mean, instruction_size in data:
                f.write(f"{architecture}:\t{auto_correlation_mean}\t{instruction_size}\n")
