import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from .data_storage import DataStorage
from .prediction_model import PredictionModel
from .resource_profiler import profiled


class PredictionHorizon:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self._tmp = df.copy()

    def __str__(self) -> str:
        return self.df.__str__()

    @property
    def start(self) -> datetime:
        return self.df.index.min().to_pydatetime()

    @property
    def end(self) -> datetime:
        return self.df.index.max().to_pydatetime()

    def in_prediction_horizon(self, dt: datetime) -> bool:
        """Whether the provided datetime is within the prediction horizon."""
        return self.start <= dt <= self.end

    def get_prediction_at(self, dt: datetime) -> pd.Series:
        """Returns the interpolated prediction for the specified datetime or a ValueError if out of range.
        The pandas.Series is indexed by the output features."""
        if not self.in_prediction_horizon(dt):
            raise ValueError(f'Datetime {dt} is out of range of prediction horizon {self.start} - {self.end}')

        try:
            return self._tmp.loc[dt]
        except KeyError:
            self._tmp.loc[dt] = np.nan
            self._tmp.interpolate(method='time', inplace=True)
            return self._tmp.loc[dt]


class Predictor:
    """Uses a PredictionModel to provide predictions for arbitrary timestamps in the prediction range of the model."""

    def __init__(self, model: PredictionModel, prediction_period: timedelta, data: DataStorage = None) -> None:
        self._model = model
        self._data = data if data is not None \
            else DataStorage(model.metadata.input_features, model.metadata.output_features)
        self._prediction_period = timedelta(seconds=prediction_period.seconds)
        self._prediction_horizon: Optional[PredictionHorizon] = None

    @property
    def model_metadata(self):
        return self._model.metadata

    @property
    def model_id(self):
        return self._model.metadata.model_id

    @property
    def data(self) -> DataStorage:
        return self._data

    @property
    def prediction_horizon_start(self):
        return self._prediction_horizon.start

    @property
    def prediction_horizon_end(self):
        return self._prediction_horizon.end

    @property
    def prediction_period(self):
        return self._prediction_period

    def set_model(self, other: PredictionModel, start: datetime):
        """Changes the underlying model of the predictor and resets the prediction horizon to the specified datetime."""
        logging.debug(f'Changing predictor model from "{self._model.metadata.model_id}" to "{other.metadata.model_id}"')
        self._model = other
        self._prediction_horizon = None
        self.update_prediction_horizon(start)

    def add_measurement(self, dt: datetime, values: np.ndarray):
        self._data.add_measurement(dt, values)

    def add_measurement_df(self, df: pd.DataFrame):
        self._data.add_measurement_df(df)

    def log_prediction(self, dt: datetime, values: np.ndarray):
        self._data.add_prediction(self.model_id, dt, values)

    def add_prediction_df(self, df: pd.DataFrame):
        self._data.add_prediction_df(df)

    def get_measurements_in_current_prediction_horizon(self, until: datetime) -> Optional[pd.DataFrame]:
        """Returns the measurements in the current horizon until the specified datetime (inclusive)."""
        if self._prediction_horizon is None:
            return None
        else:
            measurements = self._data.get_measurements_between(self.prediction_horizon_start, until)
            return measurements

    def get_reduced_measurements_in_current_prediction_horizon(self, until: datetime) -> Optional[pd.DataFrame]:
        """Returns the reduced measurements in the current horizon until the specified datetime (inclusive)."""
        if self._prediction_horizon is None:
            return None
        else:
            since = self.prediction_horizon_start
            measurements = self._data.get_reduced_measurements(since, until, self._prediction_period)
            return measurements

    def get_measurement(self, dt: datetime) -> Optional[pd.DataFrame]:
        """
        :param dt: The datetime of the measurement to retrieve.
        :return: A pandas.DataFrame containing the measurements at the specified datetime, or None if no measurement is
        available.
        """
        return self._data.get_measurements_between(dt, dt)

    def get_measurements_since(self, dt_start: datetime) -> pd.DataFrame:
        """
        :return: A pandas.DataFrame containing all the node's measurements since the specified datetime (inclusive).
        """
        return self._data.get_measurements_since(dt_start)

    def get_predictions(self) -> pd.DataFrame:
        """
        :return: A pandas.DataFrame containing all the node's predictions.
        """
        return self._data.get_predictions()

    def get_prediction_horizon(self) -> pd.DataFrame:
        """
        :return: A pandas.DataFrame containing all the values of the node's Prediction Horizon.
        """
        return self._data.get_horizon()

    def get_model_activity(self) -> pd.DataFrame:
        """
        :return: A pandas.DataFrame containing all the node's active models over time.
        """
        return self._data.get_model_activity()

    def get_predictions_until(self, until: datetime) -> Optional[pd.DataFrame]:
        if self._prediction_horizon is None:
            return None
        else:
            # the first interpolation point is a measurement, not a prediction
            return self._prediction_horizon.df.between_time(self.prediction_horizon_start.time(), until.time(),
                                                            inclusive='right'
                                                            )

    def get_prediction_at(self, dt: datetime) -> pd.Series:
        """Returns the interpolated prediction for the specified datetime or None if it is out of range.
        The pandas.Series is indexed by the output features."""
        return self._prediction_horizon.get_prediction_at(dt)

    def in_prediction_horizon(self, dt: datetime) -> bool:
        return self._prediction_horizon.in_prediction_horizon(dt)

    def adjust_to_measurement(self, dt: datetime, measurement: np.ndarray, prediction: np.ndarray) -> None:
        """Makes the predictor aware of a threshold violation so that it can adjust future predictions."""
        if self._prediction_horizon is None:
            logging.error(f'Cannot adapt non-existing interpolation points!')
            return
        diff = [measurement[i] for i in self.model_metadata.input_to_output_indices] - prediction
        logging.debug(f'Adapting prediction at {dt} by {diff}')
        self._prediction_horizon = PredictionHorizon(self._prediction_horizon.df + diff)
        logging.debug(f'New prediction horizon: \n {self._prediction_horizon}')

    @profiled(tag="Prediction")
    def update_prediction_horizon(self, start: datetime):
        """Updates the interpolation points used for computing predictions. """
        # if self._prediction_horizon is not None and \
        #         self._prediction_horizon.df.index[0] <= start < self._prediction_horizon.df.index[1]:
        #     logging.debug(f'Skipped updating prediction horizon because nothing would change: '
        #                   f'{self._prediction_horizon.df.index[0]} <= {start} < {self._prediction_horizon.df.index[1]}'
        #                   )
        #     return

        previous_m = self._data.get_previous_measurements(start,
                                                          self._model.metadata.input_length,
                                                          self._prediction_period)
        new_horizon = self._model.predict(previous_m)

        # we also need the last measurement for interpolation
        last_ts = previous_m.index.max()
        new_horizon.loc[last_ts] = previous_m.loc[last_ts, self.model_metadata.output_features]
        new_horizon.sort_index(inplace=True)
        self._data.update_horizon(new_horizon)

        previous_ip = self._prediction_horizon
        self._prediction_horizon = PredictionHorizon(new_horizon)
        logging.debug(f'Updated prediction horizon from\n{previous_ip}\nto\n{self._prediction_horizon}')

    def get_prediction_timedelta(self):
        return self._prediction_period

    def log_violation(self, dt: datetime) -> None:
        self._data.add_violation(dt, self.model_id)
