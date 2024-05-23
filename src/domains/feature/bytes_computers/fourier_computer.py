from typing import Optional
from numpy.fft import fft
from .bytes_computer import BytesComputer


class FourierComputer(BytesComputer):
    def __init__(self, data_len: Optional[int] = None) -> None:
        self.data_len = data_len

    def compute(self, data: bytes) -> list[float]:
        fft_data = fft(list(iter(data)))
        fft_real = fft_data.real
        return_len = min(filter(None, [self.data_len, len(fft_real)]))
        return list(fft_real[:return_len])

    def get_group_name(
        self, data_identifiers: Optional[list[str]] = None, override_if_empty: str = ""
    ) -> str:
        if data_identifiers is None:
            data_identifiers = [override_if_empty]
        return "_".join(data_identifiers)

    def x_labels(self) -> Optional[list[int]]:
        return None

    def y_scale(self):
        return "symlog"

    def labels(self) -> tuple[str, str]:
        return ("Frequency", "fft")
