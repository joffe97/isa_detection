from pathlib import Path
from pandas import Series

from domains.dataset.binary_file_dataset import BinaryFileDataset
from . import Researcher


class AutoCorrelation(Researcher):
    def __init__(self, byte_read_count: int = 512, lag=1) -> None:
        super().__init__()
        self.byte_read_count = byte_read_count
        self.lag = lag

    def research(self, dataset: BinaryFileDataset):
        auto_correlations_for_architecture = dict()
        for architecture, path_strs in dataset.create_architecture_paths_mapping().items():
            paths = list(map(Path, path_strs))

            for path in paths:
                with open(path, "rb") as f:
                    data = f.read(self.byte_read_count)

                filename = path.name
                data_ints = list(iter(data))
                series = Series(data_ints)
                auto_correlation = series.autocorr(self.lag)

                auto_correlations_for_architecture.setdefault(architecture, [])
                auto_correlations_for_architecture[architecture].append((filename, auto_correlation))

        group_name = "_".join(map(str, [self.byte_read_count, self.lag]))

        file_path = self._create_result_path(group_name, dataset.identifier(), ".txt")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            for architecture, auto_correlation_tuples in auto_correlations_for_architecture.items():
                is_single_item = len(auto_correlation_tuples) == 1
                for filename, auto_correlation in auto_correlation_tuples:
                    entry_key = "_".join(
                        list(map(str, filter(None, [architecture, not is_single_item and filename])))
                    )
                    f.write(f"{entry_key}:\t{auto_correlation}\n")
