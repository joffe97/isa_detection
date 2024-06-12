from typing import Optional
from pandas import Series
import numpy as np
from .bytes_computer import BytesComputer


class AutoCorrelationComputer(BytesComputer):
    def __init__(
        self,
        lag_max: int,
        lag_min: int = 1,
        autocorr_times: int = 1,
        max_data_len_for_higher_autocorr: Optional[int] = None,
    ) -> None:
        self.lag_min = lag_min
        self.lag_max = lag_max
        self.autocorr_times = autocorr_times
        self.max_data_len_for_higher_autocorr = max_data_len_for_higher_autocorr

    @property
    def lag_range(self):
        return range(self.lag_min, self.lag_max + 1)

    @property
    def lags_str(self):
        return f"{self.lag_min}-{self.lag_max}"

    def compute(self, data: bytes) -> list[float]:
        if self.autocorr_times == 0:
            return list(map(float, data))

        data_list = list(iter(data))
        if self.autocorr_times == 1:
            data_series = Series(data_list)
            autocorr_list = list(
                data_series.autocorr(lag) for lag in self.lag_range
            )
            while np.isnan(autocorr_list[-1]):  # type: ignore
                autocorr_list.pop()
            return autocorr_list

        possible_data_lens = [len(data_list)]
        if self.max_data_len_for_higher_autocorr is not None:
            possible_data_lens.append(self.max_data_len_for_higher_autocorr)
        data_len_for_higher_autocorr = min(*possible_data_lens)

        for _ in range(self.autocorr_times):
            data_list = list(
                Series(data_list).autocorr(lag)
                for lag in range(data_len_for_higher_autocorr + 1)
            )
        return data_list[self.lag_min : self.lag_max + 1]  # type: ignore

    def get_group_name(
        self,
        data_identifiers: Optional[list[str]] = None,
        override_if_empty: str = "",
    ) -> str:
        if data_identifiers is None:
            data_identifiers = [override_if_empty]
        return "_".join(data_identifiers)

    def identifier(self) -> str:
        autocorr_str = "AutoCorrelation"
        if self.autocorr_times != 1:
            autocorr_str += str(self.autocorr_times)
        if self.max_data_len_for_higher_autocorr is not None:
            autocorr_str += f"_{self.max_data_len_for_higher_autocorr}"
        return f"{autocorr_str}_{self.lags_str}"

    def x_labels(self) -> list[int]:
        return list(self.lag_range)

    def labels(self) -> tuple[str, str]:
        return ("Lag", "Autocorrelation")
