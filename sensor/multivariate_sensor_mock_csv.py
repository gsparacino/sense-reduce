import logging
import os
import random

import pandas as pd

from abstract_sensor import AbstractSensor


class MultivariateCsvMockSensor(AbstractSensor):
    """Mocks a multivariate sensor with pre-defined data."""
    BASEDIR = os.path.abspath(os.path.dirname(__file__))

    def __init__(self, csv_path: str = None) -> None:
        super().__init__()
        # TODO this list should always match the BaseStation model's input layer, make the coupling more explicit?
        #  i.e. the Sensor could receive an array of features upon registration to the BS
        self.features = ['TL', 'P', 'RF', 'SO', 'RR', 'DD']
        if csv_path:
            path = os.path.join(self.BASEDIR, csv_path)
            logging.debug(f"Loading sensor mock data {path}")
            df = pd.read_csv(path)
            df.reset_index(inplace=True)
            self.data = df
            self.iterator = df.iterrows()
        else:
            raise ValueError("CsvMockSensor requires a valid path ta a .csv file as input")

    @property
    def measurement(self) -> pd.Series:
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
