import datetime
from typing import Any

from common.model_metadata import ModelMetadata
from common.predictor_adaptation_goal import PredictorAdaptationGoal


class SensorKnowledgeUpdate:

    def __init__(self,
                 next_sync_dt: datetime.datetime,
                 adaptation_goals: list[PredictorAdaptationGoal],
                 models_portfolio: set[str]
                 ):
        self.next_sync_dt = next_sync_dt
        self.predictor_adaptation_goals = adaptation_goals
        self.models_portfolio = models_portfolio

    def to_dict(self) -> dict[str, Any]:
        predictor_goals = [goal.to_dict() for goal in self.predictor_adaptation_goals]
        return {
            "predictor_adaptation_goals": predictor_goals,
            "models_portfolio": list(self.models_portfolio)
        }


class SensorKnowledgeInitialization(SensorKnowledgeUpdate):

    def __init__(self,
                 node_id: str,
                 base_model_metadata: ModelMetadata,
                 adaptation_goals: list[PredictorAdaptationGoal],
                 models_portfolio: set[str],
                 next_sync_dt: datetime.datetime,
                 ):
        super().__init__(next_sync_dt, adaptation_goals, models_portfolio)
        self.node_id = node_id
        self.base_model_metadata = base_model_metadata

    def to_dict(self) -> dict[str, Any]:
        predictor_goals = [goal.to_dict() for goal in self.predictor_adaptation_goals]
        return {
            'node_id': self.node_id,
            'base_model_metadata': self.base_model_metadata.to_dict(),
            'predictor_adaptation_goals': predictor_goals,
            'models_portfolio': list(self.models_portfolio),
            'next_sync_dt': self.next_sync_dt.isoformat()
        }
