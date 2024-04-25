import statistics
import sys
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(1, "/home/joachan/isa_detection/src")

from config import Config
from domains.dataset import IsaDetect, CpuRec, IsaDetectCode
from domains.dataset.custom.sinus_signal import SinusSignal, CustomDataset
from researchers import (
    Researcher,
    AutoCorrelation,
    AutoCorrelationOld,
    DataVisualizer,
    AutoCorrelationMean,
    AutoCorrelationMeanPlot,
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
    lag_count = 16
    # datasets = [CpuRec(), IsaDetectCode(100)]
    # datasets = [IsaDetectCode(5)]
    datasets = [CpuRec()]
    # datasets = [SinusSignal()]
    for dataset in datasets:
        # DataVisualizer().research(dataset)
        # for i in lags:
        #     print(f"{dataset.identifier()} {i}")
        #     AutoCorrelation(max_read, i).research(dataset)
        #     AutoCorrelationMean(max_read, i).research(dataset)
        # lags = list(range(1, lag_count + 1))
        # AutoCorrelationMeanPlot(MAX_READ, lags, not isinstance(dataset, CustomDataset)).research(dataset)
        # pd.plotting.autocorrelation_plot(pd.Series(list(iter(dataset.create_data())))[0:1024])
        variable_archs_count = 0
        variable_styles = ["--", "-.", ":"]
        for arch, data_iters_generator in dataset.iter_architectures_with_files_data(64000):
            # data_iter = next(data_iters_generator)

            line_style = (
                "-"
                if arch in ["amd64", "i386"]
                else variable_styles[(variable_archs_count // 10) % len(variable_styles)]
            )
            variable_archs_count += int(line_style != "-")

            # tmp_ax = pd.plotting.autocorrelation_plot(
            #     pd.Series(list(next(data_iters_generator)))[0:32], linewidth=0.2, linestyle=line_style
            # )

            # autocorr_datas = [
            #     my_autocorr(pd.Series(list(data_iter)))[0:128] for data_iter in data_iters_generator
            # ]

            autocorr_times = 2
            autocorr_datas = [pd.Series(list(data_iter)) for data_iter in data_iters_generator]
            lag_range = (1, 17)
            for _ in range(autocorr_times):
                autocorr_datas = [
                    pd.Series(autocorr_plotfunc(autocorr_input_data))
                    for autocorr_input_data in autocorr_datas
                ]

            mean = [
                statistics.fmean(
                    [
                        autocorr_data.array[index]
                        for autocorr_data in autocorr_datas
                        if index < len(autocorr_data)
                    ]
                )
                for index in range(lag_range[1] - 1)
            ]
            plt.plot(
                list(range(*lag_range)),
                mean,
                linewidth=0.2,
                linestyle=line_style,
                label=arch,
            )
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 0.0),
            ncol=8,
            fontsize="xx-small",
        )
        plt.savefig(Config.RESEARCH_PATH.joinpath("tmp.png"), dpi=600)
        plt.close()

        # AutoCorrelationOld(2, 10240).research(dataset)
