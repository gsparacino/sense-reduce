import datetime

import pandas as pd

from sensor.abstract_sensor import AbstractSensor


class MultivariateSensorMock(AbstractSensor):

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self._idx = 0

    def can_read_measurements(self) -> bool:
        return self._idx < len(self.df)

    @property
    def measurement(self) -> (datetime.datetime, pd.Series):
        # Read measurement
        dt: datetime.datetime = self.df.index[self._idx]
        measurement: pd.Series = self.df.loc[dt]
        self._idx += 1
        return dt, measurement
