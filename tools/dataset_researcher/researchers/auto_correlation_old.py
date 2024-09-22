from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from . import FileResearcher


class AutoCorrelationOld(FileResearcher):
    def __init__(self, n_gram_size: int, byte_read_count: int = 512) -> None:
        super().__init__()
        self.byte_read_count = byte_read_count
        self.n_gram_size = n_gram_size

    def _research_file(self, architecture: str, path: Path, group_name: str):
        with open(path, "rb") as f:
            data = f.read(self.byte_read_count)

        matches_count = np.zeros(len(data), dtype=int)
        for i in range(len(data)):
            end_i = i + self.n_gram_size
            if end_i >= len(matches_count):
                continue
            cur_i_n_gram = data[i:end_i]
            for j in range(len(data)):
                end_j = j + self.n_gram_size
                if end_j >= len(matches_count):
                    continue
                cur_j_n_gram = data[j:end_j]
                if all(
                    i_byte == j_byte
                    for i_byte, j_byte in zip(cur_i_n_gram, cur_j_n_gram)
                ):
                    matches_count[i] += 1

        matches_count_fft = np.fft.fft(matches_count)
        matches_count_fft_real, matches_count_fft_imag = list(
            zip(
                *map(
                    lambda complex: (complex.real, complex.imag),
                    matches_count_fft,
                )
            )
        )

        fig, axs = plt.subplots(3)
        line_width = 0.3
        axs[0].plot(
            list(range(len(matches_count))), matches_count, linewidth=line_width
        )
        axs[1].plot(
            list(range(len(matches_count_fft_real))),
            list(map(abs, map(float, matches_count_fft_real))),
            linewidth=line_width,
        )
        axs[2].plot(
            list(range(len(matches_count_fft_imag))),
            list(map(abs, map(float, matches_count_fft_imag))),
            linewidth=line_width,
        )

        plot_path = self._create_result_path_with_architecture_and_binary_file(
            group_name, architecture, path.name, ".eps"
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=300)
        plt.close()
