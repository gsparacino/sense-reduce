import random

import pandas as pd
from pandas import DataFrame

from abstract_sensor import AbstractSensor


class MockSensor(AbstractSensor):
    """Mocks a temperature sensor with random data."""

    def __init__(self, data: DataFrame = None) -> None:
        super().__init__()
        # TODO this list should always match the BaseStation model's input layer, make the coupling more explicit?
        #  i.e. the Sensor could receive an array of features upon registration to the BS
        self.features = ["TL"]
        if data is not None:
            self.data = data
            self.iterator = data.iterrows()

    @property
    def measurement(self) -> pd.Series:
        # TODO support multivariate measurements
        if self.data is None:
            return pd.Series(
                data=[self.temperature],
                index=self.features
            )
        else:
            try:
                return self.nextItem()
            except StopIteration:
                self.iterator = self.data.iterrows()
                return self.nextItem()

    def nextItem(self):
        return next(self.iterator)[1][self.features]

    @property
    def temperature(self) -> float:
        return random.random() * 10 + 10  # interval [10, 20]

    @property
    def humidity(self) -> float:
        return random.random() * 30 + 60  # interval [60, 90]
