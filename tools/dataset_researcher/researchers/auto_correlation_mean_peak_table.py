import pandas

from domains.dataset.binary_file_dataset import BinaryFileDataset
from domains.feature.bytes_computers import AutoCorrelationComputer
from domains.label.label_entry import LabelEntry
from domains.label.label_loaders.corpus_labels import CorpusLabels
from .researcher import Researcher


class AutoCorrelationMeanPeakTable(Researcher):
    def __init__(self, byte_read_count: int, lag_max: int, lag_min: int = 1) -> None:
        self.byte_read_count = byte_read_count
        self.lag_max = lag_max
        self.lag_min = lag_min

    def research(self, dataset: BinaryFileDataset):
        autocorr_computer = AutoCorrelationComputer(self.lag_max, self.lag_min)
        architecture_file_datas_mapping = dataset.create_architecture_func_data_mapping(
            self.byte_read_count, autocorr_computer.compute
        )
        mean_datas = architecture_file_datas_mapping.mean_datas()

        labels = CorpusLabels(
            included_labels={LabelEntry.INSTRUCTION_SIZE}
        ).load_as_architecture_labels_mapping()
        architecture_data_mapping: dict[str, list[str]] = dict()
        columns: list[str] = [
            *list(
                f"{col_f_str}_{i}"
                for col_f_str in ["autocorr_peak", "autocorr_peak_bits"]
                for i in range(1, 4)
            ),
            LabelEntry.INSTRUCTION_SIZE.value,
        ]
        for architecture, mean_data in sorted(mean_datas.items(), key=lambda item: item[0].lower()):
            mean_data_sorted_by_value = sorted(
                enumerate(mean_data, autocorr_computer.lag_min),
                key=lambda enumerated_value: enumerated_value[1],
                reverse=True,
            )
            mean_data_peaks = [str(mean_data_peak) for mean_data_peak, _ in mean_data_sorted_by_value[:3]]
            mean_data_peak_bits = [
                str(mean_data_peak * 8) for mean_data_peak, _ in mean_data_sorted_by_value[:3]
            ]
            instruction_size = str(labels.get(architecture, dict()).get(LabelEntry.INSTRUCTION_SIZE))
            architecture_data_mapping[architecture] = [
                *mean_data_peaks,
                *mean_data_peak_bits,
                instruction_size,
            ]

        dataframe = pandas.DataFrame.from_dict(architecture_data_mapping, orient="index", columns=columns)
        pandas.set_option("display.max_rows", dataframe.shape[0] + 1)

        group_name = "_".join(map(str, [self.byte_read_count, f"{self.lag_min}-{self.lag_max}"]))
        file_name = dataset.identifier()
        path = self._create_result_path(group_name, file_name, ".txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(str(dataframe))
