from pathlib import Path
from typing import Any, Optional

from plotters.plotter import Plotter
from domains.dataset.classes.architecture_file_datas_mapping import (
    ArchitectureFileDatasMapping,
)
from domains.dataset.binary_file_dataset import BinaryFileDataset
from domains.label.label_entry import LabelEntry
from domains.feature.bytes_computers import BytesComputer
from .researcher import Researcher


class BytesComputerPlotter(Researcher):
    def __init__(
        self, byte_read_count: int, bytes_computer: BytesComputer
    ) -> None:
        super().__init__()
        self.byte_read_count = byte_read_count
        self.bytes_computer = bytes_computer

        self.__group_name = None

    def get_group_name(self):
        if self.__group_name is None:
            self.__group_name = self.bytes_computer.get_group_name(
                [str(self.byte_read_count)]
            )
        return self.__group_name

    def get_group_name_for_architecture(self, architecture: str) -> str:
        return f"{self.get_group_name()}/architecture_data/{architecture}"

    def __class_name_override(self) -> str:
        bytes_computer_identifier = self.bytes_computer.identifier()
        return f"{bytes_computer_identifier}MeanPlot"

    def __plotter(self, **kwargs) -> Plotter:
        x_label, y_label = self.bytes_computer.labels()
        plotter_args: dict[str, Any] = {
            "yscale": self.bytes_computer.y_scale(),
            "xlabel": x_label,
            "ylabel": y_label,
        }
        plotter_args.update(kwargs)
        return Plotter(**plotter_args)

    def _plot_file_data_mappings(
        self,
        architecture_data_mapping: ArchitectureFileDatasMapping,
    ):
        class_name_override = self.__class_name_override()

        file_data_mappings = architecture_data_mapping.file_data_mappings()
        architecture_data_means_mapping = architecture_data_mapping.mean_datas()
        plotter = self.__plotter(dpi=200)
        for architecture, file_data_tuples in file_data_mappings.items():
            architecture_group_name = self.get_group_name_for_architecture(
                architecture
            )
            for file_name, file_data in file_data_tuples:
                plot_path = self._create_result_path(
                    architecture_group_name,
                    file_name,
                    ".png",
                    class_name_override=class_name_override,
                )
                plotter.plot(
                    self.bytes_computer.x_labels(),
                    {file_name: file_data},
                    f"{architecture} - {file_name}",
                    plot_path,
                )
            if architecture_mean_data := architecture_data_means_mapping.get(
                architecture
            ):
                plot_path = self._create_result_path(
                    architecture_group_name,
                    "_mean",
                    ".png",
                    class_name_override=class_name_override,
                )
                plotter.plot(
                    self.bytes_computer.x_labels(),
                    {f"{architecture} mean": architecture_mean_data},
                    f"{architecture} - Mean",
                    plot_path,
                )

    def research(self, dataset: BinaryFileDataset):
        architecture_data_mapping = (
            dataset.create_architecture_func_data_mapping(
                self.byte_read_count,
                self.bytes_computer.compute,
            )
        )

        architecture_data_means_mapping = architecture_data_mapping.mean_datas()
        architecture_data_means_group_mapping = dict(
            (architecture, str(instruction_size))
            for architecture in architecture_data_means_mapping.keys()
            if (
                architecture_file_datas := architecture_data_mapping.get(
                    architecture
                )
            )
            and (
                instruction_size := architecture_file_datas.find_label_value(
                    LabelEntry.INSTRUCTION_SIZE
                )
            )
            is not None
        )

        instruction_type_data_means_mapping = (
            architecture_data_mapping.get_means_grouped_by_label_entry(
                LabelEntry.IS_VARIABLE_INSTRUCTION_SIZE,
                label_mapping={True: "Variable", False: "Fixed"},
            )
        )

        instruction_width_data_means_mapping = (
            architecture_data_mapping.get_means_grouped_by_label_entry(
                LabelEntry.FIXED_INSTRUCTION_SIZE
            )
        )

        group_name = self.get_group_name()
        class_name_override = self.__class_name_override()

        def create_file_path_func(
            filename_ending: Optional[str] = None,
        ) -> Path:
            filename = "_".join(
                filter(None, [dataset.identifier(), filename_ending])
            )
            return self._create_result_path(
                group_name,
                filename,
                ".png",
                class_name_override=class_name_override,
            )

        self.__plotter(dpi=800).plot(
            self.bytes_computer.x_labels(),
            architecture_data_means_mapping,
            dataset.identifier(),
            create_file_path_func(),
            line_group_mapping=architecture_data_means_group_mapping,
        )

        mean_plotter = self.__plotter()
        mean_plotter.plot(
            self.bytes_computer.x_labels(),
            instruction_type_data_means_mapping,
            f"{dataset.identifier()} - Type means",
            create_file_path_func("typemean"),
        )

        mean_plotter.plot(
            self.bytes_computer.x_labels(),
            instruction_width_data_means_mapping,
            f"{dataset.identifier()} - Size means",
            create_file_path_func("sizemean"),
        )

        self._plot_file_data_mappings(architecture_data_mapping)
