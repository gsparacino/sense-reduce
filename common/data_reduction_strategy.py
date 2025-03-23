import datetime
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from common.data_storage import DataStorage
from common.model_metadata import ModelMetadata


class DataReductionStrategy(ABC):

    @abstractmethod
    def get_measurements_for_prediction(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            until: datetime.datetime
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_horizon_update_data(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            dt: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def get_violation_data(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            dt: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        pass

    def get_new_measurements(self, data, dt, model_metadata):
        since: Optional[datetime.datetime] = data.last_synchronization_dt
        if since is None:
            since = self._datetime_to_full_hour(dt) - datetime.timedelta(hours=model_metadata.input_length)
        since = self._datetime_to_full_hour(since)
        elapsed_hours = int((dt - since).total_seconds() / 3600) + 1
        return data.get_measurements_previous_hours(dt, elapsed_hours)

    @staticmethod
    def _datetime_to_full_hour(dt: datetime) -> datetime:
        if dt.minute > 0 or dt.second > 0 or dt.microsecond > 0:
            dt = dt + datetime.timedelta(hours=1)
        dt = dt.replace(minute=0, second=0, microsecond=0)
        return dt


class DownsamplingStrategy(DataReductionStrategy):

    def get_measurements_for_prediction(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            until: datetime.datetime
    ) -> pd.DataFrame:
        length = model_metadata.input_length
        return data.get_measurements_previous_hours(until, length)

    def get_horizon_update_data(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            dt: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        return self.get_new_measurements(data, dt, model_metadata)

    def get_violation_data(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            dt: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        return self.get_new_measurements(data, dt, model_metadata)


class DualPredictionStrategy(DataReductionStrategy):

    def get_measurements_for_prediction(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            until: datetime.datetime
    ) -> pd.DataFrame:
        length = model_metadata.input_length
        measurements: pd.DataFrame = data.get_measurements_previous_hours(until, length)
        predictions: pd.DataFrame = data.get_predictions_previous_hours(until, length)
        if predictions is not None:
            measurements.update(predictions)
        return measurements

    def get_horizon_update_data(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            dt: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        measurements = self.get_new_measurements(data, dt, model_metadata)
        columns = self.get_columns_to_transmit(model_metadata)
        return measurements[columns]

    def get_violation_data(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            dt: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        measurements = self.get_new_measurements(data, dt, model_metadata)
        columns = self.get_columns_to_transmit(model_metadata)
        return measurements[columns]

    @staticmethod
    def get_columns_to_transmit(model_metadata: ModelMetadata) -> list[str]:
        return [c for c in model_metadata.input_features if c not in model_metadata.output_features]
