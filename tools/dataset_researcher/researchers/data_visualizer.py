from pathlib import Path
from typing import Optional
from matplotlib import pyplot as plt
from . import FileResearcher


class DataVisualizer(FileResearcher):
    def __init__(self, max_read: Optional[int]) -> None:
        self.max_read = max_read

    def _research_file(self, architecture: str, path: Path, group_name: str):
        with open(path, "rb") as f:
            data = f.read(self.max_read)

        data_ints = list(map(float, data))

        line_width = 0.3
        plt.plot(data_ints, linewidth=line_width)

        plot_path = self._create_result_path_with_architecture_and_binary_file(
            group_name, architecture, path.name, ".eps"
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()
