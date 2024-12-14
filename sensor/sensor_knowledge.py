import datetime
from typing import Optional

import pandas as pd

from common.data_reduction_strategy import DataReductionStrategy
from common.data_storage import DataStorage
from common.model_metadata import ModelMetadata
from common.predictor import Predictor
from common.sensor_adaptation_goal import SensorAdaptationGoal
from common.sensor_knowledge_update import SensorKnowledgeUpdate, SensorKnowledgeInitialization
from sensor.base_station import BaseStation
from sensor.model_manager import ModelManager


class SensorKnowledge:

    def __init__(
            self,
            node_id: str,
            base_station: BaseStation,
            # initial_df: pd.DataFrame,
            data_reduction_strategy: DataReductionStrategy,
            model_dir: str
    ):
        self.node_id = node_id
        self.predictor: Optional[Predictor] = None
        self.adaptation_goals: list[SensorAdaptationGoal] = []
        self.base_station = base_station
        self.base_model_metadata: Optional[ModelMetadata] = None
        # self.initial_df: pd.DataFrame = initial_df
        self.model_manager: ModelManager = ModelManager(model_dir, base_station.gateway)
        self.pending_adaptation: bool = False
        self.next_adaptation_dt: Optional[datetime.datetime] = None
        self.data_reduction_strategy: DataReductionStrategy = data_reduction_strategy

    @property
    def configuration_id(self) -> str:
        return self.predictor.model_metadata.uuid

    def update(self, dt: datetime.datetime, data: SensorKnowledgeUpdate) -> None:
        self.next_adaptation_dt = data.next_adaptation_dt
        if data.predictor_adaptation_goals is not None:
            self.adaptation_goals = data.predictor_adaptation_goals

        if data.models_portfolio is not None and len(data.models_portfolio) > 0:
            new_models = self.model_manager.synchronize_models(data.models_portfolio)
            for model_id in new_models:
                self.predictor.data.add_model_deployment(dt, model_id)

    def synchronization_enabled(self, dt: datetime.datetime) -> bool:
        return False if self.next_adaptation_dt is None else dt >= self.next_adaptation_dt

    def synchronize_configurations(self, dt: datetime.datetime) -> set[str]:
        old_models = set(self.model_manager.get_models().keys())
        sync = self.base_station.sync(dt)
        self.update(dt, sync)
        updated_models = set(self.model_manager.get_models().keys())
        new_models = updated_models.difference(old_models)
        for model_id in new_models:
            self.predictor.data.add_model_deployment(dt, model_id)
        return new_models

    @classmethod
    def from_initialization(cls,
                            base_station: BaseStation,
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
        result.model_manager.synchronize_models(initial_models)
        metadata = result.base_model_metadata
        base_model = result.model_manager.add_model(metadata)
        data_storage = DataStorage(metadata.input_features, metadata.output_features)
        for model_id in initial_models:
            data_storage.add_model_deployment(last_dt, model_id)
        data_storage.add_measurement_df(initial_df)
        data_storage.add_configuration_update(last_dt, base_model.metadata.uuid)
        data_storage.last_synchronization_dt = last_dt
        result.predictor = Predictor(base_model, data_storage, data_reduction_strategy)
        # result.predictor.update_prediction_horizon(last_dt)
        return result
