import datetime
import logging
from typing import Optional

import pandas as pd

from common.data_reduction_strategy import DataReductionStrategy
from common.data_storage import DataStorage
from common.model_metadata import ModelMetadata
from common.predictor import Predictor
from common.predictor_adaptation_goal import PredictorAdaptationGoal
from common.sensor_knowledge_update import SensorKnowledgeUpdate, SensorKnowledgeInitialization
from sensor.base_station_adapter import BaseStationAdapter
from sensor.model_portfolio import ModelPortfolio


class SensorKnowledge:

    def __init__(
            self,
            node_id: str,
            base_station: BaseStationAdapter,
            data_reduction_strategy: DataReductionStrategy,
            model_dir: str
    ):
        self.node_id = node_id
        self.predictor: Optional[Predictor] = None
        self.adaptation_goals: list[PredictorAdaptationGoal] = []
        self.base_station = base_station
        self.base_model_metadata: Optional[ModelMetadata] = None
        self.model_portfolio: ModelPortfolio = ModelPortfolio(model_dir, base_station.gateway)
        self.pending_adaptation: bool = False
        self.next_sync_dt: Optional[datetime.datetime] = None
        self.data_reduction_strategy: DataReductionStrategy = data_reduction_strategy

    @property
    def configuration_id(self) -> str:
        return self.predictor.model_metadata.uuid

    def update(self, dt: datetime.datetime, update: SensorKnowledgeUpdate) -> None:
        self.next_sync_dt = update.next_sync_dt
        if update.predictor_adaptation_goals is not None:
            self.adaptation_goals = update.predictor_adaptation_goals

        if update.models_portfolio is not None and len(update.models_portfolio) > 0:
            new_models = self.model_portfolio.synchronize_models(update.models_portfolio)
            for model_id in new_models:
                self.predictor.data.add_model_deployment(dt, model_id)

    def synchronization_enabled(self, dt: datetime.datetime) -> bool:
        # return False if self.next_sync_dt is None else dt >= self.next_sync_dt
        return True

    def synchronize_configurations(self, dt: datetime.datetime) -> set[str]:
        old_models = set(self.model_portfolio.get_models().keys())
        sync = self.base_station.synchronize(dt)
        self.update(dt, sync)
        updated_models = set(self.model_portfolio.get_models().keys())
        new_models = updated_models.difference(old_models)
        for model_id in new_models:
            self.predictor.data.add_model_deployment(dt, model_id)
        return new_models

    @classmethod
    def from_initialization(cls,
                            base_station: BaseStationAdapter,
                            initial_df: pd.DataFrame,
                            initialization: SensorKnowledgeInitialization,
                            data_reduction_strategy: DataReductionStrategy,
                            model_dir: str
                            ) -> 'SensorKnowledge':
        last_dt = initial_df.index.max()
        initial_models = set(initialization.models_portfolio)
        result = cls(initialization.node_id, base_station, data_reduction_strategy, model_dir)
        result.adaptation_goals = initialization.predictor_adaptation_goals
        result.base_model_metadata = initialization.base_model_metadata
        result.model_portfolio.synchronize_models(initial_models)
        metadata = result.base_model_metadata
        base_model = result.model_portfolio.add_model(metadata)
        data_storage = DataStorage(metadata.input_features, metadata.output_features)
        for model_id in initial_models:
            data_storage.add_model_deployment(last_dt, model_id)
        data_storage.add_configuration_update(last_dt, base_model.metadata.uuid)
        data_storage.last_synchronization_dt = last_dt
        data_storage.next_synchronization_dt = initialization.next_sync_dt
        result.predictor = Predictor(base_model, data_storage, data_reduction_strategy)
        result.predictor.add_measurement_df(initial_df)
        return result

    def update_prediction_horizon(self, dt: datetime):
        logging.info(f"{dt.isoformat()} Renewing prediction horizon")
        predictor = self.predictor
        predictor.update_prediction_horizon(dt)

    def adjust_predictions(self, dt: datetime.datetime, measurement: pd.Series, prediction: pd.Series):
        logging.info(f"{dt.isoformat()} Adjusting predictions to latest measurement")
        self.predictor.adjust_to_measurement(dt, measurement.to_numpy(), prediction.to_numpy())
