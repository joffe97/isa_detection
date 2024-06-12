from typing import Optional
import numpy as np
import math
from numpy.fft import fft
from domains.feature.bytes_computers.auto_correlation_computer import (
    AutoCorrelationComputer,
)
from domains.feature.bytes_computers.bytes_computer import BytesComputer


class AutoCorrelationFftComputer(BytesComputer):
    def __init__(self, result_half_len: Optional[int]) -> None:
        self.data_half_len = result_half_len

    @property
    def data_len(self) -> Optional[int]:
        return self.data_half_len * 2 + 1 if self.data_half_len else None

    def compute(self, data: bytes) -> list[int]:
        autocorr_data = AutoCorrelationComputer(len(data)).compute(data)
        fft_real = fft(autocorr_data).real

        if (data_len := self.data_len) is not None:
            missing_len = len(fft_real) - data_len
            if missing_len > 0:
                pad_width = math.ceil(missing_len / 2)
                fft_real = np.pad(fft_real, pad_width)

        fft_middle_index = len(fft_real) // 2
        return (
            fft_real[
                (fft_middle_index - self.data_half_len) : (
                    fft_middle_index + self.data_half_len + 1
                )
            ]
            if self.data_half_len is not None
            else fft_real
        )

    def get_group_name(
        self,
        data_identifiers: Optional[list[str]] = None,
        override_if_empty: str = "",
    ) -> str:
        if data_identifiers is None:
            data_identifiers = [override_if_empty]
        return "_".join(data_identifiers)

    def x_labels(self) -> Optional[list[int]]:
        if self.data_half_len is None:
            return None
        return list(range(-self.data_half_len, self.data_half_len + 1))

    def labels(self) -> tuple[str, str]:
        return ("Frequency", "Amplitude")

    def identifier(self) -> str:
        autocorr_str = "AutoCorrelationFft"
        return f"{autocorr_str}_{self.data_half_len}"
