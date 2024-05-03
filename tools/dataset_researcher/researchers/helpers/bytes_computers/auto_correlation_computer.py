from typing import Optional
from pandas import Series
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
        elif self.autocorr_times == 1:
            return list(Series(list(iter(data))).autocorr(lag) for lag in self.lag_range)

        data_list = list(iter(data))
        possible_data_lens = [len(data_list)]
        if self.max_data_len_for_higher_autocorr is not None:
            possible_data_lens.append(self.max_data_len_for_higher_autocorr)
        data_len_for_higher_autocorr = min(*possible_data_lens)
        for _ in range(self.autocorr_times):
            data_list = list(
                Series(data_list).autocorr(lag) for lag in range(data_len_for_higher_autocorr + 1)
            )
        return data_list[self.lag_min : self.lag_max + 1]  # type: ignore

    def get_group_name(self, data_identifiers: Optional[list[str]] = None) -> str:
        if data_identifiers is None:
            data_identifiers = []
        return "_".join([*data_identifiers, self.lags_str])

    def identifier(self) -> str:
        autocorr_str = "AutoCorrelation"
        if self.autocorr_times == 1:
            return autocorr_str
        return f"{autocorr_str}{self.autocorr_times}_{self.max_data_len_for_higher_autocorr}"
