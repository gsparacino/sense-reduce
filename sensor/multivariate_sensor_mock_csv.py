import logging
import os

import pandas as pd

from abstract_sensor import AbstractSensor


class MultivariateCsvMockSensor(AbstractSensor):
    BASEDIR = os.path.abspath(os.path.dirname(__file__))

    def __init__(self, csv_path: str) -> None:
        """
        A multivariate sensor that returns pre-defined data, loaded from a CSV file."

        :param csv_path: the relative path to the CSV file containing the sensor data.
        """
        super().__init__()
        # TODO this list should always match the model's input layer, make the coupling more explicit?
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
            raise ValueError("MultivariateCsvMockSensor requires a valid path ta a CSV file as input")

    @property
    def measurement(self) -> pd.Series:
        try:
            return self.nextItem()
        except StopIteration:
            self.iterator = self.data.iterrows()
            return self.nextItem()

    def nextItem(self):
        return next(self.iterator)[1][self.features]
