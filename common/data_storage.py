import datetime
import os
from typing import List

import numpy as np
import pandas as pd

from .utils import to_full_hour, timestamps_before


# TODO (long-term): this class should become a layer for accessing a time-series database (e.g., InfluxDB)
class DataStorage:
    """A wrapper around two pandas Dataframe for past measurements and predictions."""

    def __init__(self, input_features: List[str], output_features: List[str]) -> None:
        self._measurements = pd.DataFrame(columns=input_features, dtype=np.float64)
        self._predictions = pd.DataFrame(columns=output_features, dtype=np.float64)

    @property
    def mae(self) -> pd.Series:
        return self.get_diff().abs().mean()

    @property
    def mse(self) -> pd.Series:
        return (self.get_diff() ** 2).mean()

    @property
    def rmse(self) -> pd.Series:
        return self.mse ** 0.5

    @staticmethod
    def find_nearest_index(indices, value):
        idx = (np.abs(indices - value)).argmin()
        return idx

    @classmethod
    def from_data(cls, measurements: pd.DataFrame, predictions: pd.DataFrame) -> 'DataStorage':
        storage = cls(measurements.columns, predictions.columns)
        storage._measurements = measurements
        storage._predictions = predictions
        return storage

    @classmethod
    def from_previous_years_average(cls, start: datetime.datetime,
                                    end: datetime.datetime,
                                    previous: pd.DataFrame,
                                    output_features: List[str]) -> 'DataStorage':
        """Creates StorageData with measurements set in the given range using average previous values."""
        # TODO: add a parameter max_prev_years to limit the number of previous years to use
        assert end > start, "Expected end date to be after start date"
        data = DataStorage(previous.columns, output_features)
        i = to_full_hour(start)  # force full-hours
        while i <= end:
            j = i - datetime.timedelta(days=365)  # this way we don't have to consider Feb 29 separately
            previous_values = pd.DataFrame(columns=previous.columns)
            try:
                value = previous.loc[j]
            except KeyError:
                value = None
            while value is not None:
                previous_values.loc[j] = value
                j -= datetime.timedelta(days=365)
                try:
                    value = previous.loc[j]
                except KeyError:
                    value = None
            data.add_measurement(i, previous_values.mean())
            i += datetime.timedelta(hours=1)
        return data

    @classmethod
    def from_previous_average(cls,
                              days: int,
                              previous: pd.DataFrame,
                              output_features: List[str]) -> 'DataStorage':
        """Creates StorageData with measurements from the latest available data"""
        start: datetime.datetime = previous.index.min()
        end: datetime.datetime = previous.index.max()

        if (end - start).days > days:
            start = end - datetime.timedelta(days=days)

        return cls.from_previous_years_average(start, end, previous, output_features)

    @classmethod
    def csv_path_measurements(cls, prefix='') -> str:
        if prefix != '':
            return f'{prefix}_measurements.csv'
        return f'measurements.csv'

    @classmethod
    def csv_path_predictions(cls, prefix='') -> str:
        if prefix != '':
            return f'{prefix}_predictions.csv'
        return f'predictions.csv'

    def add_measurement(self, dt: datetime.datetime, values: np.ndarray):
        self._measurements.loc[dt] = values

    def add_prediction(self, dt: datetime.datetime, values: np.ndarray):
        self._predictions.loc[dt] = values

    def add_measurement_dict(self, d: dict):
        for date_string, values in d.items():
            self.add_measurement(datetime.datetime.fromisoformat(date_string), values)

    def add_prediction_dict(self, d: dict):
        for date_string, values in d.items():
            self.add_prediction(datetime.datetime.fromisoformat(date_string), values)

    def add_measurement_df(self, df: pd.DataFrame):
        self._measurements = pd.concat([self._measurements, df], copy=False)

    def add_prediction_df(self, df: pd.DataFrame):
        self._predictions = pd.concat([self._predictions, df], copy=False)

    def copy(self, deep=True) -> 'DataStorage':
        """Returns a deep copy of this object. Note that the csv_path is also equal unless changed afterwards."""
        copy = DataStorage(self._measurements.columns, self._predictions.columns)
        copy._measurements = self._measurements.copy(deep)
        copy._predictions = self._predictions.copy(deep)
        return copy

    def get_measurements(self) -> pd.DataFrame:
        return self._measurements

    def get_predictions(self) -> pd.DataFrame:
        return self._predictions

    def get_previous_measurements(self,
                                  until_dt: datetime.datetime,
                                  number_of_measurements: int,
                                  timedelta: datetime.timedelta
                                  ) -> pd.DataFrame:
        """
        Returns the measurements within the provided time span, which starts at (until_dt) - (timedelta * number_of_measurements)
        and ends at (until_dt).
        If the sensor does not have enough measurements in the provided timeframe, older measurements are translated to
        fill the missing timestamps, and added to the results.

        :param until_dt: the timestamp of the latest measurement to include in the results
        :param number_of_measurements: the number of measurements to include in the results
        :param timedelta: the interval between consecutive measurements in the results
        :return: a DataFrame with the measurements within the provided time span
        """
        timestamps = list(
            timestamps_before(until_dt, number_of_measurements, timedelta)
        )
        idx = [self.find_nearest_index(self._measurements.index, timestamp) for timestamp in timestamps]
        result: pd.DataFrame = self._measurements.iloc[idx].copy()
        result.set_index(pd.DatetimeIndex(timestamps), inplace=True)
        return result

    def get_measurements_between(self, dt_start: datetime.datetime, dt_end: datetime.datetime) -> pd.DataFrame:
        """
        :param dt_start: the start timestamp
        :param dt_end: the end timestamp
        :return: a DataFrame of measurements between the provided start and end timestamps.
        """
        return self._measurements.loc[dt_start:dt_end]

    def get_diff(self, columns: List[str] = None) -> pd.DataFrame:
        """Returns the difference between measurements and predictions. Removes NaNs."""
        diff: pd.DataFrame = self._measurements.loc[:, self._predictions.columns] - self._predictions
        if columns is not None:
            diff = diff[columns]
        return diff[~diff.isna().any(axis=1)]

    def save(self, dir_path='.', csv_prefix='') -> None:
        os.makedirs(dir_path, exist_ok=True)
        self._measurements.to_csv(os.path.join(dir_path, self.csv_path_measurements(csv_prefix)), index=True)
        self._predictions.to_csv(os.path.join(dir_path, self.csv_path_predictions(csv_prefix)), index=True)

    @classmethod
    def load(cls, dir_path='.', csv_prefix='') -> 'DataStorage':
        measurements = pd.read_csv(os.path.join(dir_path, DataStorage.csv_path_measurements(csv_prefix)),
                                   index_col=0, parse_dates=True)
        predictions = pd.read_csv(os.path.join(dir_path, DataStorage.csv_path_predictions(csv_prefix)),
                                  index_col=0, parse_dates=True)

        ds = DataStorage(measurements.columns, predictions.columns)
        ds._measurements = measurements
        ds._predictions = predictions
        return ds
