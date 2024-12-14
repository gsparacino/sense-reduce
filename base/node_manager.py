import datetime
from typing import Optional

from common.data_reduction_strategy import DataReductionStrategy
from common.data_storage import DataStorage
from common.prediction_model import PredictionModel
from common.predictor import Predictor
from common.sensor_adaptation_goal import SensorAdaptationGoal
from common.sensor_knowledge_update import SensorKnowledgeUpdate


class NodeManager:

    def __init__(self,
                 node_id: str,
                 input_features: list[str],
                 output_features: list[str],
                 model: PredictionModel,
                 data_reduction_strategy: DataReductionStrategy,
                 ):
        self.node_id = node_id
        self.enabled = False
        self.last_adaptation_dt: Optional[datetime.datetime] = None
        self.next_adaptation_dt: Optional[datetime.datetime] = None
        self.data = DataStorage(input_features, output_features)
        self.predictor: Predictor = Predictor(model, self.data, data_reduction_strategy)
        self._adaptation_goals: dict[str, SensorAdaptationGoal] = {}
        self._models: set[str] = set()

    def get_adaptation_goals(self) -> list[SensorAdaptationGoal]:
        return list(self._adaptation_goals.values())

    def add_adaptation_goal(self, adaptation_goal: SensorAdaptationGoal) -> None:
        self._adaptation_goals[adaptation_goal.goal_id] = adaptation_goal

    def remove_adaptation_goal(self, goal_id: str) -> None:
        self._adaptation_goals.pop(goal_id)

    def get_model_ids(self) -> set[str]:
        return self._models

    def get_active_model_id(self) -> str:
        return self.predictor.model_metadata.uuid

    def set_active_model(self, model: PredictionModel, dt: datetime.datetime) -> None:
        if self.predictor.model_metadata.uuid != model.metadata.uuid:
            self.predictor.set_model(model, dt)

    def set_models(self, models: set[str]) -> None:
        self._models = models

    def add_model(self, model: str) -> None:
        self._models.add(model)

    def remove_model(self, model: str) -> None:
        self._models.remove(model)

    def to_sensor_knowledge_update(self) -> SensorKnowledgeUpdate:
        return SensorKnowledgeUpdate(
            self.next_adaptation_dt,
            self.get_adaptation_goals(),
            self.get_model_ids()
        )
