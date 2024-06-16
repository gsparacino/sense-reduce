import datetime
import logging
import uuid

import numpy as np
import pandas as pd
import tensorflow as tf

from common import full_hours_after, convert_datetime, ModelMetadata
from common.prediction_model import PredictionModel

ModelID = uuid


class Model(PredictionModel):
    """Wrapper for a keras Sequential model and its associated metadata."""

    def __init__(self,
                 model: tf.keras.Model,
                 metadata: ModelMetadata
                 ) -> None:
        logging.debug(f'Created new model with metadata {metadata}')
        self._metadata: ModelMetadata = metadata
        self._model: tf.keras.Model = model
        self._check_model()

    @property
    def model_id(self) -> ModelID:
        return str(self.metadata.model_id)

    @property
    def model(self) -> tf.keras.Model:
        return self._model

    @property
    def metadata(self) -> ModelMetadata:
        """Returns a reference to the model's metadata. Use deepcopy() if needed."""
        return self._metadata

    def _check_model(self):
        if self.model.input_shape != self.metadata.input_shape and self.model.input_shape[0] is not None:
            logging.warning(f'Model input shape and metadata input shape do not match, '
                            f'{self.model.input_shape} != {self.metadata.input_shape}')
        if self.model.output_shape != self.metadata.output_shape and self.model.output_shape[0] is not None:
            logging.warning(f'Model output shape and metadata output shape do not match, '
                            f'{self.model.output_shape} != {self.metadata.output_shape}')

    def predict(self, x: pd.DataFrame) -> pd.DataFrame:
        """Runs a single inference of the model.

        Args:
            x: A pandas.DataFrame with DatetimeIndex

        Returns:
            A pandas DataFrame with DatetimeIndex
        """
        convert_datetime(x, self.metadata.periodicity)
        x_np = self._normalize_input_features(x).values.reshape((1, self.metadata.input_length, -1))

        return self._model_output_to_dataframe(self.model.predict(x_np), x.index.max_ts())

    def _normalize_input_features(self, x: pd.DataFrame) -> pd.DataFrame:
        x[self.metadata.input_features] = (x[self.metadata.input_features] - self.metadata.input_normalization_mean) \
                                          / self.metadata.input_normalization_std
        return x

    def _model_output_to_dataframe(self, a: np.ndarray, dt: datetime.datetime) -> pd.DataFrame:
        a = (a * self.metadata.output_normalization_std) + self.metadata.output_normalization_mean
        return pd.DataFrame(data=a.reshape((self.metadata.output_length, self.metadata.output_attributes)),
                            columns=self.metadata.output_features,
                            index=full_hours_after(dt, self.metadata.output_length))

    def get_parameter_count(self) -> int:
        return self.model.count_params()
