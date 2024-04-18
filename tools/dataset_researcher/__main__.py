import sys

sys.path.insert(1, "/home/joachan/isa_detection/src")

from domains.dataset import IsaDetect, CpuRec, IsaDetectCode
from domains.dataset.custom.sinus_signal import SinusSignal
from researchers import (
    Researcher,
    AutoCorrelation,
    AutoCorrelationOld,
    DataVisualizer,
    AutoCorrelationMean,
    AutoCorrelationMeanPlot,
)

if __name__ == "__main__":
    MAX_READ = 64000
    lag_count = 128
    # datasets = [CpuRec(), IsaDetectCode(100)]
    datasets = [IsaDetectCode(5)]
    # datasets = [SinusSignal()]
    for dataset in datasets:
        # DataVisualizer().research(dataset)
        # for i in lags:
        #     print(f"{dataset.identifier()} {i}")
        #     AutoCorrelation(max_read, i).research(dataset)
        #     AutoCorrelationMean(max_read, i).research(dataset)
        lags = list(range(1, lag_count + 1))
        AutoCorrelationMeanPlot(MAX_READ, lags, False).research(dataset)
        # AutoCorrelationOld(2, 10240).research(dataset)
