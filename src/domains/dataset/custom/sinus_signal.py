import numpy as np
import math

from domains.dataset.custom.custom_dataset import CustomDataset


class SinusSignal(CustomDataset):
    # def create_data(self) -> bytes:
    #     n = 1000
    #     time_array = np.linspace(0, 1000, 1000 * 10 + 1)
    #     sinus = 100 * np.sin(time_array) + 100
    #     sinus_bytes = bytes(map(int, sinus))
    #     return sinus_bytes

    def create_data(self) -> bytes:
        amplitude = 100
        period = 100
        num_points = 10000
        """
        Generates a sine wave with the given amplitude and period.

        Args:
        amplitude (int): The amplitude of the sine wave.
        period (int): The period of the sine wave.
        num_points (int): The number of points to generate.

        Returns:
        list: A list of integers representing the sine wave.
        """
        sine_values = []
        for t in range(num_points):
            # Calculate the sine value at time t
            sine_value = (
                amplitude * math.sin(2 * math.pi * t / period) + amplitude
            )
            # Convert the sine value to an integer
            sine_values.append(int(sine_value))
        return bytes(sine_values)
