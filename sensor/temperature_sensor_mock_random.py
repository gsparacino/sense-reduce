import random

import pandas as pd

from abstract_sensor import AbstractSensor


class RandomMockSensor(AbstractSensor):
    """Mocks a temperature sensor with random data."""

    def __init__(self) -> None:
        super().__init__()
        # TODO this list should always match the BaseStation model's input layer, make the coupling more explicit?
        #  i.e. the Sensor could receive an array of features upon registration to the BS
        self.features = ["TL"]

    @property
    def measurement(self) -> pd.Series:
        # TODO support multivariate measurements
        return pd.Series(
            data=[self.temperature],
            index=self.features
        )

    @property
    def temperature(self) -> float:
        return random.random() * 10 + 10  # interval [10, 20]

    @property
    def humidity(self) -> float:
        return random.random() * 30 + 60  # interval [60, 90]
