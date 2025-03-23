import datetime
import logging
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import full_hours_before, to_full_hour

CONFIGURATION_ID_COLUMN = "configuration_id"
EVALUATION_COLUMN = "evaluation"
DT_COLUMN = "timestamp"
MODEL_UUID_COLUMN = "model_uuid"


# TODO (long-term): this class should become a layer for accessing a time-series database (e.g., InfluxDB)
class DataStorage:

    def __init__(self, input_features: List[str], output_features: List[str]) -> None:
        self.last_synchronization_dt: Optional[datetime.datetime] = None
        self.next_synchronization_dt: Optional[datetime.datetime] = None
        self._measurements = pd.DataFrame(columns=input_features, dtype=np.float64)
        self._predictions = pd.DataFrame(columns=output_features, dtype=np.float64)
        self._violations = pd.DataFrame(columns=[MODEL_UUID_COLUMN], dtype=np.float64)
        self._horizon_updates = pd.DataFrame(columns=[MODEL_UUID_COLUMN])
        self._model_deployments = pd.DataFrame(columns=[DT_COLUMN, MODEL_UUID_COLUMN])
        self._configuration_updates = pd.DataFrame(columns=[CONFIGURATION_ID_COLUMN])
        self._analysis_performed = pd.DataFrame(columns=[DT_COLUMN, CONFIGURATION_ID_COLUMN, EVALUATION_COLUMN])

    @property
    def mae(self) -> pd.Series:
        return self.get_diff().abs().mean()

    @property
    def mse(self) -> pd.Series:
        return (self.get_diff() ** 2).mean()

    @property
    def rmse(self) -> pd.Series:
        return self.mse ** 0.5

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
            previous_idx = previous.index.get_indexer([j], method='nearest')
            value = previous.iloc[previous_idx]
            while value is not None:
                previous_values = previous_values.append(value)
                j -= datetime.timedelta(days=365)
                try:
                    value = previous.loc[j]
                except KeyError:
                    value = None
            data.add_measurement(i, previous_values.mean())
            i += datetime.timedelta(hours=1)
        return data

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

    def add_violation(self, dt: datetime.datetime, model: str):
        self._violations.loc[dt] = model

    def get_violations(self) -> pd.DataFrame:
        return self._violations

    def get_latest_consecutive_violations_of_model(self, model: str, until: datetime.datetime) -> pd.DataFrame:
        violations: pd.DataFrame = self.get_violations().loc[:until]
        violations_of_prev_model: pd.DataFrame = violations[violations[MODEL_UUID_COLUMN] != model]
        if not violations_of_prev_model.empty:
            last_violation_of_prev_model = violations_of_prev_model.index.max()
            start_timestamp = last_violation_of_prev_model + pd.Timedelta(seconds=1)
            return violations.loc[start_timestamp:]
        else:
            return violations

    def add_horizon_update(self, dt: datetime.datetime, model: str):
        self._horizon_updates.loc[dt] = model

    def get_horizon_updates(self) -> pd.DataFrame:
        return self._horizon_updates

    def add_configuration_update(self, dt: datetime.datetime, option_id: str):
        self._configuration_updates.loc[dt] = option_id

    def get_configuration_updates(self) -> pd.DataFrame:
        return self._configuration_updates

    def add_model_deployment(self, dt: datetime.datetime, model_id: str):
        new_deployment = {MODEL_UUID_COLUMN: model_id, DT_COLUMN: dt}
        idx = len(self._model_deployments)
        self._model_deployments.loc[idx] = new_deployment

    def get_model_deployments(self) -> pd.DataFrame:
        return self._model_deployments

    def add_analysis(self, dt: datetime.datetime, option_id: str, evaluation: str):
        self._analysis_performed.loc[len(self._analysis_performed)] = [dt, option_id, evaluation]

    def get_analysis(self) -> pd.DataFrame:
        return self._analysis_performed

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
        self._measurements = pd.concat([self._measurements, df], copy=False).groupby(level=0).last()
        self._measurements.sort_index(inplace=True)

    def add_prediction_df(self, df: pd.DataFrame):
        self._predictions = pd.concat([self._predictions, df], copy=False).groupby(level=0).last()
        self._predictions.sort_index(inplace=True)

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

    def get_predictions_previous_hours(self, dt: datetime.datetime, n_hours: int) -> Optional[pd.DataFrame]:
        full_hours = list(full_hours_before(dt, n_hours))
        return self._get_nearest_predictions(full_hours)

    def get_prediction(self, dt: datetime.datetime) -> pd.Series:
        return self._predictions.loc[dt]

    def get_measurements_previous_hours(self, dt: datetime.datetime, n_hours: int) -> pd.DataFrame:
        """Returns the measurements at the full hours before the specified timestamp (inclusive).

        If there are no measurements for a full hour, the values of the next one are used.
        """
        full_hours = list(full_hours_before(dt, n_hours))  # will result in an already sorted list
        return self._get_nearest_measurements(full_hours)

    def _get_nearest_measurements(self, dts: list[datetime.datetime]):
        logging.debug(f"Loading measurements with timestamps [{[hour.isoformat() for hour in dts]}]")
        idx = self._measurements.index.get_indexer(dts, method='nearest')
        result: pd.DataFrame = self._measurements.iloc[idx].copy()
        result.set_index(pd.DatetimeIndex(dts), inplace=True)
        logging.debug(f"Size of measurements: {result.size}")
        return result

    def _get_nearest_predictions(self, dts: list[datetime.datetime]) -> Optional[pd.DataFrame]:
        if self._predictions.empty:
            return None
        logging.debug(f"Loading predictions with timestamps [{[hour.isoformat() for hour in dts]}]")
        idx = self._predictions.index.get_indexer(dts, method='nearest')
        result: pd.DataFrame = self._predictions.iloc[idx].copy()
        result.set_index(pd.DatetimeIndex(dts), inplace=True)
        logging.debug(f"Size of predictions: {result.size}")
        return result

    def get_diff(self, columns: List[str] = None) -> pd.DataFrame:
        """Returns the difference between measurements and predictions. Removes NaNs."""
        diff: pd.DataFrame = self._measurements.loc[:, self._predictions.columns] - self._predictions
        if columns is not None:
            diff = diff[columns]
        return diff[~diff.isna().any(axis=1)]

    def plot(self):
        """Creates a plot for every attribute, comparing measurements and predictions."""
        for col in self._measurements.columns:
            plt.plot(self._measurements[col], label='Measurement')
            plt.plot(self._predictions[col], label='Prediction')
            plt.show()

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
