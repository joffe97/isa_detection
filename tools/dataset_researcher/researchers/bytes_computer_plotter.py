from pathlib import Path
from typing import Optional

from plotters.plotter import Plotter
from domains.dataset.classes.architecture_file_datas_mapping import ArchitectureFileDatasMapping
from domains.dataset.binary_file_dataset import BinaryFileDataset
from domains.label.label_entry import LabelEntry
from .researcher import Researcher
from .helpers.bytes_computers.bytes_computer import BytesComputer
from .helpers.bytes_computers.auto_correlation_computer import AutoCorrelationComputer


class BytesComputerPlotter(Researcher):
    def __init__(self, byte_read_count: int, bytes_computer: BytesComputer) -> None:
        super().__init__()
        self.byte_read_count = byte_read_count
        self.bytes_computer = bytes_computer

        self.__lags_str = None
        self.__group_name = None

    def lags_str(self):
        if self.__lags_str is None:
            is_continous = True
            lags_set = set(self.lags)
            min_lag = min(lags_set)
            max_lag = max(lags_set)
            for lag in range(min_lag, max_lag + 1):
                if lag not in lags_set:
                    is_continous = False
                    break

            if is_continous:
                self.__lags_str = f"{min_lag}-{max_lag}"
            else:
                self.__lags_str = "+".join(map(str, self.lags))

        return self.__lags_str

    def get_group_name(self):
        if self.__group_name is None:
            self.__group_name = self.bytes_computer.get_group_name([str(self.byte_read_count)])
        return self.__group_name

    def get_group_name_for_architecture(self, architecture: str) -> str:
        return f"{self.get_group_name()}/architecture_data/{architecture}"

    @property
    def class_name_suffix(self) -> Optional[str]:
        return (
            str(auto_correlation_count)
            if (auto_correlation_count := self.auto_correlation_count) != 1
            else None
        )

    def _plot_file_data_mappings(
        self,
        architecture_data_mapping: ArchitectureFileDatasMapping,
    ):
        file_data_mappings = architecture_data_mapping.file_data_mappings()
        architecture_data_means_mapping = architecture_data_mapping.mean_datas()
        dpi = 200
        for architecture, file_data_tuples in file_data_mappings.items():
            architecture_group_name = self.get_group_name_for_architecture(architecture)
            for file_name, file_data in file_data_tuples:
                plot_path = self._create_result_path(
                    architecture_group_name, file_name, ".png", class_name_suffix=self.class_name_suffix
                )
                Plotter().plot(
                    self.lags, {file_name: file_data}, f"{architecture} - {file_name}", plot_path, dpi=dpi
                )
            if architecture_mean_data := architecture_data_means_mapping.get(architecture):
                plot_path = self._create_result_path(
                    architecture_group_name, "_mean", ".png", class_name_suffix=self.class_name_suffix
                )
                Plotter().plot(
                    self.lags,
                    {f"{architecture} mean": architecture_mean_data},
                    f"{architecture} - Mean",
                    plot_path,
                    dpi=dpi,
                )

    def research(self, dataset: BinaryFileDataset):
        architecture_data_mapping = dataset.create_architecture_func_data_mapping(
            self.byte_read_count,
            self.bytes_computer.compute,
        )

        architecture_data_means_mapping = architecture_data_mapping.mean_datas()
        architecture_data_means_group_mapping = dict(
            (architecture, str(instruction_size))
            for architecture in architecture_data_means_mapping.keys()
            if (architecture_file_datas := architecture_data_mapping.get(architecture))
            and (instruction_size := architecture_file_datas.find_label_value(LabelEntry.INSTRUCTION_SIZE))
            is not None
        )

        instruction_type_data_means_mapping = architecture_data_mapping.get_means_grouped_by_label_entry(
            LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
            label_mapping={True: "Variable", False: "Fixed"},
        )

        instruction_width_data_means_mapping = architecture_data_mapping.get_means_grouped_by_label_entry(
            LabelEntry.INSTRUCTION_SIZE
        )

        group_name = self.get_group_name()

        def create_file_path_func(filename_ending: Optional[str] = None) -> Path:
            filename = "_".join(filter(None, [dataset.identifier(), filename_ending]))
            return self._create_result_path(
                group_name, filename, ".png", class_name_suffix=self.class_name_suffix
            )

        plotter = Plotter()
        plotter.plot(
            self.lags,
            architecture_data_means_mapping,
            dataset.identifier(),
            create_file_path_func(),
            line_group_mapping=architecture_data_means_group_mapping,
            dpi=800,
        )

        plotter.plot(
            self.lags,
            instruction_type_data_means_mapping,
            f"{dataset.identifier()} - Type means",
            create_file_path_func("typemean"),
        )

        plotter.plot(
            self.lags,
            instruction_width_data_means_mapping,
            f"{dataset.identifier()} - Size means",
            create_file_path_func("sizemean"),
        )

        self._plot_file_data_mappings(architecture_data_mapping)
