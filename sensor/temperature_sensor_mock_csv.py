import logging
import random

import pandas as pd

from abstract_sensor import AbstractSensor


class CsvMockSensor(AbstractSensor):
    """Mocks a temperature sensor with random data."""

    def __init__(self, csvPath: str = None) -> None:
        super().__init__()
        # TODO this list should always match the BaseStation model's input layer, make the coupling more explicit?
        #  i.e. the Sensor could receive an array of features upon registration to the BS
        self.features = ["TL"]
        if csvPath:
            logging.debug(f"Loading sensor mock data {csvPath}")
            df = pd.read_csv(csvPath)
            df.reset_index(inplace=True)
            self.data = df
            self.iterator = df.iterrows()
        else:
            raise ValueError("CsvMockSensor requires a valid path ta a .csv file as input")

    @property
    def measurement(self) -> pd.Series:
        # TODO support multivariate measurements
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
