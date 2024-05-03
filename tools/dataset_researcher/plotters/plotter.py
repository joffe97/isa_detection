import math
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt

ALL_LINE_STYLES = ["-", "dashdot", "dotted", "dashed", (0, (3, 5, 1, 5, 1, 5))]


class Plotter:
    def plot(
        self,
        lines_x: list[int],
        lines_y: dict[str, list[float]],
        title: str,
        path: Path,
        line_group_mapping: Optional[dict[str, str]] = None,
        dpi: int = 500,
    ):
        if line_group_mapping is None:
            line_group_mapping = dict((label, label) for label in lines_y.keys())
        line_groups = set(line_group_mapping.values())

        fig, ax = plt.subplots(figsize=(6.5, 8))
        fig.subplots_adjust(bottom=0.25)
        # mean_axs_instruction_size_replacer = {}
        # if self.include_means:
        #     fig1, ax1 = plt.subplots()
        #     fig2, ax2 = plt.subplots(figsize=(6.5, 8))
        #     fig2.subplots_adjust(bottom=0.25)
        #     axs_list.extend([ax1, ax2])
        #     mean_axs_instruction_size_replacer.update({"type_mean": ax1, "size_mean": ax2})

        lines_count = len(lines_y)
        line_width = next(
            (
                corr_line_width
                for max_ax_sixe, corr_line_width in [(5, 1.0), (10, 0.7), (20, 0.4)]
                if lines_count <= max_ax_sixe
            ),
            0.2,
        )

        all_line_styles_extended = ALL_LINE_STYLES * (math.ceil(len(line_groups) / len(ALL_LINE_STYLES)))
        line_group_line_style_cycle_mapping = dict(
            (
                line_group,
                all_line_styles_extended[
                    (len(all_line_styles_extended) * line_group_index // len(line_groups)) : (
                        len(all_line_styles_extended) * (line_group_index + 1) // len(line_groups)
                    )
                ],
            )
            for line_group_index, line_group in enumerate(
                sorted(
                    line_groups,
                    key=lambda line_group: list(line_group_mapping.values()).count(line_group),
                )
            )
        )
        # 5 * 0 // 3 = 0
        # 5 * 1 // 3 = 1
        # 5 * 2 // 3 = 3
        # 5 * 3 // 3 = 5
        line_group_traversed_counts = dict((line_style_group, 0) for line_style_group in line_groups)
        for label, line_data in lines_y.items():
            line_style = "-"
            line_group = line_group_mapping.get(label)
            if line_group is not None:
                line_style_cycle = line_group_line_style_cycle_mapping[line_group]
                line_style = line_style_cycle[
                    (line_group_traversed_counts[line_group] // 10) % len(line_style_cycle)
                ]

            if line_group is not None and label != line_group:
                label = ", ".join([label, line_group])

            ax.plot(lines_x, line_data, label=label, linewidth=line_width, linestyle=line_style)

        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=4,
            fontsize="xx-small",
        )
        ax.set_title(title)

        # group_name = "_".join(map(str, [self.byte_read_count, self.lags_str()]))
        # file_path = self._create_result_path(group_name, dataset.identifier(), ".png")

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=dpi)
        plt.close(fig)

        # if self.include_means:
        #     ax1.plot(
        #         self.lags,
        #         np.array(data_with_all_lags["Fixed mean"][0])
        #         - np.array(data_with_all_lags["Variable mean"][0]),
        #         label="Difference",
        #         linewidth=line_width[ax1],
        #         linestyle="--",
        #     )
        #     ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 0.1), ncol=3)
        #     ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4)
        #     ax1.set_title(f"{dataset.identifier()} - Type means (absolute)")
        #     ax2.set_title(f"{dataset.identifier()} - Size means (absolute)")

        #     file_path_type_mean = self._create_result_path(
        #         group_name, f"{dataset.identifier()}_typemean", ".png"
        #     )
        #     file_path_size_mean = self._create_result_path(
        #         group_name, f"{dataset.identifier()}_sizemean", ".png"
        #     )
        #     fig1.savefig(str(file_path_type_mean), dpi=500)
        #     fig2.savefig(str(file_path_size_mean), dpi=500)
