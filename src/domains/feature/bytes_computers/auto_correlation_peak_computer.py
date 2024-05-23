from typing import Optional
from domains.feature.bytes_computers.auto_correlation_computer import AutoCorrelationComputer
from domains.feature.bytes_computers.bytes_computer import BytesComputer


class AutoCorrelationPeakComputer(BytesComputer):
    def __init__(self, lag_max: int, lag_min: int = 1, n_peaks: int = 1) -> None:
        self.lag_min = lag_min
        self.lag_max = lag_max
        self.n_peaks = n_peaks

    @property
    def lag_range(self):
        return range(self.lag_min, self.lag_max + 1)

    @property
    def lags_str(self):
        return f"{self.lag_min}-{self.lag_max}"

    def compute(self, data: bytes) -> list[int]:
        autocorr_data = AutoCorrelationComputer(self.lag_max, self.lag_min).compute(data)
        enumerated_autocorr_data_sorted_descending = sorted(
            enumerate(autocorr_data), key=lambda x: x[1], reverse=True
        )
        enumerated_autocorr_data_sorted_descending_largest = enumerated_autocorr_data_sorted_descending[
            : self.n_peaks
        ]
        return [index for index, _ in enumerated_autocorr_data_sorted_descending_largest]

    def get_group_name(
        self, data_identifiers: Optional[list[str]] = None, override_if_empty: str = ""
    ) -> str:
        if data_identifiers is None:
            data_identifiers = [override_if_empty]
        return "_".join(data_identifiers)

    def x_labels(self) -> Optional[list[int]]:
        return list(range(self.n_peaks))

    def labels(self) -> tuple[str, str]:
        return ("Value", "Peak value")

    def identifier(self) -> str:
        autocorr_str = "AutoCorrelationPeak"
        if self.n_peaks != 1:
            autocorr_str += str(self.n_peaks)
        return f"{autocorr_str}_{self.lags_str}"
