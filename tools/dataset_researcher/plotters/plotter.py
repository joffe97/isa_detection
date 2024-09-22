import math
from pathlib import Path
from typing import Literal, Optional
import matplotlib.pyplot as plt
import numpy as np

ALL_LINE_STYLES = ["-", "dashdot", "dotted", "dashed", (0, (3, 5, 1, 5, 1, 5))]


class Plotter:
    def __init__(
        self,
        yscale: Literal["linear", "log", "symlog", "logit"] = "linear",
        dpi: int = 500,
        xlabel: str = "",
        ylabel: str = "",
    ) -> None:
        self.yscale = yscale
        self.dpi = dpi
        self.xlabel = xlabel
        self.ylabel = ylabel

    def plot(
        self,
        lines_x: Optional[list[int]],
        lines_y: dict[str, list[float]],
        title: str,
        path: Path,
        line_group_mapping: Optional[dict[str, str]] = None,
    ):
        if line_group_mapping is None:
            line_group_mapping = dict(
                (label, label) for label in lines_y.keys()
            )
        line_groups = set(line_group_mapping.values())

        fig, ax = plt.subplots(figsize=(6.5, 8))
        fig.subplots_adjust(bottom=0.25)

        lines_count = len(lines_y)
        line_width = next(
            (
                corr_line_width
                for max_ax_sixe, corr_line_width in [
                    (5, 1.0),
                    (10, 0.7),
                    (20, 0.4),
                ]
                if lines_count <= max_ax_sixe
            ),
            0.2,
        )

        all_line_styles_extended = ALL_LINE_STYLES * (
            math.ceil(len(line_groups) / len(ALL_LINE_STYLES))
        )
        line_group_line_style_cycle_mapping = dict(
            (
                line_group,
                all_line_styles_extended[
                    (
                        len(all_line_styles_extended)
                        * line_group_index
                        // len(line_groups)
                    ) : (
                        len(all_line_styles_extended)
                        * (line_group_index + 1)
                        // len(line_groups)
                    )
                ],
            )
            for line_group_index, line_group in enumerate(
                sorted(
                    line_groups,
                    key=lambda line_group: list(
                        line_group_mapping.values()
                    ).count(line_group),
                )
            )
        )

        line_group_traversed_counts = dict(
            (line_style_group, 0) for line_style_group in line_groups
        )
        for label, line_data in sorted(lines_y.items()):
            line_style = "-"
            line_group = line_group_mapping.get(label)
            if line_group is not None:
                line_style_cycle = line_group_line_style_cycle_mapping[
                    line_group
                ]
                line_style = line_style_cycle[
                    (line_group_traversed_counts[line_group] // 10)
                    % len(line_style_cycle)
                ]

            if line_group is not None and label != line_group:
                label = " ".join(map(str, [label, line_group]))

            if lines_x is None:
                ax.plot(
                    line_data,
                    label=label,
                    linewidth=line_width,
                    linestyle=line_style,
                )
            else:
                ax.plot(
                    lines_x,
                    line_data,
                    label=label,
                    linewidth=line_width,
                    linestyle=line_style,
                )

        if self.xlabel:
            ax.set_xlabel(self.xlabel)
        if self.ylabel:
            ax.set_ylabel(self.ylabel)

        ax.set_xticks(np.arange(0, 33, 4))
        ax.set_yscale(self.yscale)  # type: ignore

        legend_ncol = 4 + int(len(lines_y) > 30)
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.1),
            ncol=legend_ncol,
            fontsize="small",
        )
        ax.set_title(title)

        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=self.dpi)
        plt.close(fig)
