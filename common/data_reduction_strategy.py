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
    def on_horizon_update(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            dt: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def on_violation(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            dt: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        pass

    @abstractmethod
    def synchronization_enabled(self, data: DataStorage, dt: datetime.datetime) -> bool:
        pass

    @staticmethod
    def _datetime_to_full_hour(dt: datetime) -> datetime:
        if dt.minute > 0 or dt.second > 0 or dt.microsecond > 0:
            dt = dt + datetime.timedelta(hours=1)
        dt = dt.replace(minute=0, second=0, microsecond=0)
        return dt


class NoReductionStrategy(DataReductionStrategy):

    def get_measurements_for_prediction(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            until: datetime.datetime
    ) -> pd.DataFrame:
        length = model_metadata.input_length
        return data.get_measurements_previous_hours(until, length)

    def on_horizon_update(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            dt: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        since: Optional[datetime.datetime] = data.last_synchronization_dt
        if since is None:
            since = self._datetime_to_full_hour(dt) - datetime.timedelta(hours=model_metadata.input_length)
        since = self._datetime_to_full_hour(since)
        elapsed_hours = int((dt - since).total_seconds() / 3600) + 1
        return data.get_measurements_previous_hours(dt, elapsed_hours)

    def on_violation(
            self,
            data: DataStorage,
            model_metadata: ModelMetadata,
            dt: datetime.datetime
    ) -> Optional[pd.DataFrame]:
        since: Optional[datetime.datetime] = data.last_synchronization_dt
        if since is None:
            since = self._datetime_to_full_hour(dt) - datetime.timedelta(hours=model_metadata.input_length)
        since = self._datetime_to_full_hour(since)
        elapsed_hours = int((dt - since).total_seconds() / 3600) + 1
        return data.get_measurements_previous_hours(dt, elapsed_hours)

    def synchronization_enabled(self, data: DataStorage, dt: datetime.datetime) -> bool:
        return True
