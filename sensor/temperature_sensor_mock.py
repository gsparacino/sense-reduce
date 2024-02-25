import random

import pandas as pd

from abstract_sensor import AbstractSensor


class MockSensor(AbstractSensor):
    """Mocks a temperature sensor with random data."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def measurement(self) -> pd.Series:
        # TODO return values of other features as well
        return pd.Series(data=[0, 0, 0, 0, 0, 0, self.temperature, 0, 0, 0],
                         index=["DD", "FFAM", "P", "RF", "RR", "SO", "TL", "DD_sin", "DD_cos", "RR_norm"])

    @property
    def temperature(self) -> float:
        return random.random() * 10 + 10  # interval [10, 20]

    @property
    def humidity(self) -> float:
        return random.random() * 30 + 60  # interval [60, 90]
