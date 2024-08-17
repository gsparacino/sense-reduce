import random

import pandas as pd

from abstract_sensor import AbstractSensor


class TemperatureSensor(AbstractSensor):
    """Mocks a temperature sensor with random data."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def measurement(self) -> pd.Series:
        data = [
            self.temperature,
            self.pressure,
            self.humidity,
            self.sunshine
        ]
        return pd.Series(data=data, index=['TL', 'P', 'RF', 'SO'])

    @property
    def temperature(self) -> float:
        return random.random() * 10 + 10  # interval [10, 20]

    @property
    def pressure(self) -> float:
        return random.random() * 50 + 950  # interval [950, 1000]

    @property
    def humidity(self) -> float:
        return random.random() * 30 + 60  # interval [60, 90]

    @property
    def sunshine(self) -> float:
        return random.random() * 600  # interval [0, 600]
