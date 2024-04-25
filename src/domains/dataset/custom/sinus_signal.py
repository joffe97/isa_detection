import numpy as np

from domains.dataset.custom.custom_dataset import CustomDataset


class SinusSignal(CustomDataset):
    def create_data(self) -> bytes:
        time_array = np.linspace(0, 1280, 1280 * 10 + 1)
        sinus = 100 * np.sin(time_array) + 100
        sinus_bytes = bytes(map(int, sinus))
        return sinus_bytes
