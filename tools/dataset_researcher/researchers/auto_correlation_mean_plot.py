import matplotlib.pyplot as plt
import numpy as np

from domains.dataset.binary_file_dataset import BinaryFileDataset
from domains.dataset.custom.custom_dataset import CustomDataset
from . import Researcher, AutoCorrelationMean


class AutoCorrelationMeanPlot(Researcher):
    def __init__(
        self,
        byte_read_count: int,
        lags: list[int],
        include_means: bool = True,
        auto_correlation_count: int = 1,
    ) -> None:
        super().__init__()
        self.byte_read_count = byte_read_count
        self.lags = lags
        self.include_means = include_means
        self.auto_correlation_count = auto_correlation_count

    def lags_str(self):
        is_continous = True
        lags_set = set(self.lags)
        min_lag = min(lags_set)
        max_lag = max(lags_set)
        for lag in range(min_lag, max_lag + 1):
            if lag not in lags_set:
                is_continous = False
                break

        if is_continous:
            return f"{min_lag}-{max_lag}"
        else:
            return "+".join(map(str, self.lags))

    def research(self, dataset: BinaryFileDataset):
        data_with_all_lags = dict()
        for lag in self.lags:
            auto_correlation_mean = AutoCorrelationMean(self.byte_read_count, lag)
            auto_correlation_mapping = auto_correlation_mean.get_auto_correlation_mapping(dataset)
            data = AutoCorrelationMean.get_auto_correlation_means(
                auto_correlation_mapping,
                self.include_means,
                include_architectures_without_instruction_size=isinstance(dataset, CustomDataset),
            )
            for architecture, autocorrelation, instruction_size in data:
                data_with_all_lags.setdefault(architecture, ([], instruction_size))
                data_with_all_lags[architecture][0].append(autocorrelation)

        fig0, ax0 = plt.subplots(figsize=(6.5, 8))
        fig0.subplots_adjust(bottom=0.25)
        axs_list = [ax0]
        mean_axs_instruction_size_replacer = {}
        if self.include_means:
            fig1, ax1 = plt.subplots()
            fig2, ax2 = plt.subplots(figsize=(6.5, 8))
            fig2.subplots_adjust(bottom=0.25)
            axs_list.extend([ax1, ax2])
            mean_axs_instruction_size_replacer.update({"type_mean": ax1, "size_mean": ax2})

        ax_sizes = dict(
            (
                cur_ax,
                len(
                    [
                        cur_instruction_size
                        for (_, cur_instruction_size) in data_with_all_lags.values()
                        if (mean_axs_instruction_size_replacer.get(cur_instruction_size, ax0) == cur_ax)
                    ]
                ),
            )
            for cur_ax in axs_list
        )
        line_width_mapping = dict(
            (
                cur_ax,
                next(
                    (
                        corr_line_width
                        for max_ax_sixe, corr_line_width in [(5, 1.0), (10, 0.7), (20, 0.4)]
                        if ax_sizes[cur_ax] <= max_ax_sixe
                    ),
                    0.2,
                ),
            )
            for cur_ax in axs_list
        )

        instruction_type_counts_for_axs = dict((cur_ax, {True: 0, False: 0}) for cur_ax in axs_list)
        instruction_type_line_style_cycle_mapping = {
            True: ["-", "dashdot"],
            False: ["dotted", "dashed", (0, (3, 5, 1, 5, 1, 5))],
        }
        for architecture, (autocorrelations, instruction_size) in data_with_all_lags.items():
            cur_ax = mean_axs_instruction_size_replacer.get(instruction_size, ax0)

            is_mean = "mean" in instruction_size
            if (not self.include_means) and is_mean:
                continue

            line_width = line_width_mapping[cur_ax]

            is_variable_instruction_size = "-" in instruction_size or cur_ax != ax0
            line_style_cycle = instruction_type_line_style_cycle_mapping[is_variable_instruction_size]
            line_style = line_style_cycle[
                (instruction_type_counts_for_axs[cur_ax][is_variable_instruction_size] // 10)
                % len(line_style_cycle)
            ]
            instruction_type_counts_for_axs[cur_ax][is_variable_instruction_size] += 1

            cur_ax.plot(
                self.lags,
                autocorrelations,
                label=", ".join(filter(None, [architecture, instruction_size if not is_mean else None])),
                linewidth=line_width,
                linestyle=line_style,
            )

        for cur_ax in axs_list:
            cur_ax.set_xlabel("Lag")
            cur_ax.set_ylabel("Autocorrelation")

        ax0.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=4,
            fontsize="xx-small",
        )
        ax0.set_title(dataset.identifier())

        group_name = "_".join(map(str, [self.byte_read_count, self.lags_str()]))
        file_path = self._create_result_path(group_name, dataset.identifier(), ".png")
        file_path.parent.mkdir(parents=True, exist_ok=True)

        fig0.savefig(str(file_path), dpi=800)

        if self.include_means:
            ax1.plot(
                self.lags,
                np.array(data_with_all_lags["Fixed mean"][0])
                - np.array(data_with_all_lags["Variable mean"][0]),
                label="Difference",
                linewidth=line_width_mapping[ax1],
                linestyle="--",
            )
            ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3)
            ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
            ax1.set_title(f"{dataset.identifier()} - Type means (absolute)")
            ax2.set_title(f"{dataset.identifier()} - Size means (absolute)")

            file_path_type_mean = self._create_result_path(
                group_name, f"{dataset.identifier()}_typemean", ".png"
            )
            file_path_size_mean = self._create_result_path(
                group_name, f"{dataset.identifier()}_sizemean", ".png"
            )
            fig1.savefig(str(file_path_type_mean), dpi=500)
            fig2.savefig(str(file_path_size_mean), dpi=500)
