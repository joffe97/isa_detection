import sys
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


sys.path.insert(1, "/home/joachan/isa_detection/src")
sys.path.insert(1, "/home/joachan/isa_detection/tools/dataset_researcher")

from .researchers.helpers.bytes_computers.auto_correlation_computer import AutoCorrelationComputer

from config import Config
from domains.dataset import IsaDetect, CpuRec, IsaDetectCode
from domains.dataset.custom.sinus_signal import SinusSignal, CustomDataset
from researchers import (
    Researcher,
    AutoCorrelation,
    AutoCorrelationOld,
    DataVisualizer,
    AutoCorrelationMean,
    BytesComputerPlotter,
)


def my_autocorr(x):
    result = np.correlate(x, x, mode="full")
    return result[result.size // 2 :]


def autocorr_plotfunc(x, lag_range: Optional[tuple[int, int]] = None) -> np.ndarray:
    AUTOCORR_LINE_INDEX = 5

    fig, ax = plt.subplots()
    pd.plotting.autocorrelation_plot(x, ax=ax)
    [_, y] = ax.lines[AUTOCORR_LINE_INDEX].get_data()
    plt.close(fig)

    return y[lag_range[0] - 1 : min(lag_range[1] - 1, len(y))] if lag_range is not None else y


if __name__ == "__main__":
    MAX_READ = 64000
    # MAX_READ = 1200
    # MAX_READ = 20000
    lag_count = 16
    # datasets = [CpuRec(), IsaDetectCode(100)]
    # datasets = [IsaDetectCode(5), CpuRec()]
    datasets = [CpuRec()]
    # datasets = [SinusSignal()]
    for dataset in datasets:
        lags = list(range(1, lag_count + 1))
        bytes_computer = AutoCorrelationComputer(
            16,
            autocorr_times=1,
            max_data_len_for_higher_autocorr=1000,
        )
        BytesComputerPlotter(MAX_READ, lags, not isinstance(dataset, CustomDataset)).research(
            dataset, bytes_computer
        )
