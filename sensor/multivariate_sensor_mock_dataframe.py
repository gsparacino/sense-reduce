import os

import pandas as pd
from pandas import DataFrame

from sensor.abstract_sensor import AbstractSensor


class MultivariateDataframeMockSensor(AbstractSensor):
    BASEDIR = os.path.abspath(os.path.dirname(__file__))

    def __init__(self, df: DataFrame) -> None:
        """
        A multivariate sensor that returns pre-defined data, loaded from a Pandas' DataFrame.

        :param data: the Dataframe containing the sensor data.
        """
        super().__init__()
        # TODO this list should always match the model's input layer, make the coupling more explicit?
        #  i.e. the Sensor could receive an array of features upon registration to the BS
        self.features = ['TL', 'P', 'RF', 'SO', 'RR', 'DD']
        self.data = df
        self.iterator = df.iterrows()

    @property
    def measurement(self) -> pd.Series:
        try:
            return self.nextItem()
        except StopIteration:
            self.iterator = self.data.iterrows()
            return self.nextItem()

    def nextItem(self):
        return next(self.iterator)[1][self.features]
