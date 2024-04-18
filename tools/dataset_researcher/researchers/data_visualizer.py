from pathlib import Path
from matplotlib import pyplot as plt
from . import FileResearcher
import struct


class DataVisualizer(FileResearcher):
    def _research_file(self, architecture: str, path: Path, group_name: str):
        with open(path, "rb") as f:
            data = f.read()

        data_ints = list(map(float, data))

        line_width = 0.3
        plt.plot(data_ints, linewidth=line_width)

        plot_path = self._create_result_path_with_architecture_and_binary_file(
            group_name, architecture, path.name, ".png"
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()
