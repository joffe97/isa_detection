from typing import Optional
import math
import numpy as np
from numpy.fft import fft
from .bytes_computer import BytesComputer


class FourierComputer(BytesComputer):
    def __init__(self, data_half_len: Optional[int] = None) -> None:
        self.data_half_len = data_half_len

    @property
    def data_len(self) -> Optional[int]:
        return self.data_half_len * 2 + 1 if self.data_half_len else None

    def compute(self, data: bytes) -> list[float]:
        fft_abs = list(np.abs(fft(list(iter(data)))))

        if self.data_half_len is None:
            return fft_abs

        missing_len = len(fft_abs) - self.data_len  # type: ignore
        if missing_len > 0:
            pad_width = math.ceil(missing_len / 2)
            fft_abs = np.pad(fft_abs, pad_width)

        fft_middle_index = len(fft_abs) // 2
        return fft_abs[
            (fft_middle_index - self.data_half_len) : (
                fft_middle_index + self.data_half_len + 1
            )
        ]

    def get_group_name(
        self,
        data_identifiers: Optional[list[str]] = None,
        override_if_empty: str = "",
    ) -> str:
        if data_identifiers is None:
            data_identifiers = [override_if_empty]
        return "_".join(data_identifiers)

    def identifier(self) -> str:
        identifiers = ["Fourier"]
        if self.data_half_len is not None:
            identifiers.append(str(self.data_half_len))
        return "".join(identifiers)

    def x_labels(self) -> Optional[list[int]]:
        if self.data_half_len is None:
            return None
        return list(range(-self.data_half_len, self.data_half_len + 1))

    def y_scale(self):
        return "symlog"

    def labels(self) -> tuple[str, str]:
        return ("Frequency", "Amplitude")
