from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from pandas import Series, DataFrame

from base.model import Model, ModelID
from common import ThresholdMetric, PredictionHorizon, DataStorage

NodeID = str


class NodeManager:
    """Manages a sensor node."""

    FILE_NAME = 'node_manager.json'

    def __init__(self,
                 node_id: NodeID,
                 threshold_metric: ThresholdMetric,
                 model: Model
                 ):
        self.node_id: NodeID = node_id
        self.threshold_metric: ThresholdMetric = threshold_metric
        self.model: Optional[Model] = model
        self._data_storage: DataStorage = DataStorage(model.metadata.input_features, model.metadata.output_features)
        self._prediction_horizon: Optional[PredictionHorizon] = None

    def get_prediction_at(self, dt: datetime) -> Series:
        if (self._prediction_horizon is None
                or not self._prediction_horizon.in_prediction_horizon(dt)
        ):
            self._update_prediction_horizon(dt)
        return self._prediction_horizon.get_prediction_at(dt)

    def add_measurements(self, measurements: pd.DataFrame):
        self._data_storage.add_measurement_df(measurements)

    def get_measurements_between(self, start: datetime, end: datetime) -> DataFrame:
        return self._data_storage.get_measurements_between(start, end)

    def get_measurements(self) -> DataFrame:
        return self._data_storage.get_measurements()

    def add_violation(self, dt: datetime, model_id: ModelID):
        self._data_storage.add_violation(dt, model_id)

    def get_violations_of_model(self, model_id: ModelID) -> DataFrame:
        violations = self._data_storage.get_violations()
        return violations.loc[violations['model'] == model_id]

    def _update_prediction_horizon(self, dt: datetime) -> None:
        dt_start = dt - timedelta(minutes=1)
        data = self._data_storage.get_measurements_between(dt_start, dt)
        predictions = self.model.predict(data)
        last_ts = data.index.max_ts()
        predictions.loc[last_ts] = data.loc[last_ts, self.model.metadata.output_features]
        predictions.sort_index(inplace=True)
        self._prediction_horizon = PredictionHorizon(predictions)
