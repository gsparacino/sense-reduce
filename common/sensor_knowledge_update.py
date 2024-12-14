import datetime
from typing import Any

import pandas as pd

from common.model_metadata import ModelMetadata
from common.sensor_adaptation_goal import SensorAdaptationGoal


class SensorKnowledgeUpdate:

    def __init__(self,
                 next_update_dt: datetime.datetime,
                 adaptation_goals: list[SensorAdaptationGoal],
                 models_portfolio: set[str]
                 ):
        self.next_adaptation_dt = next_update_dt
        self.predictor_adaptation_goals = adaptation_goals
        self.models_portfolio: set[str] = models_portfolio

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
                 adaptation_goals: list[SensorAdaptationGoal],
                 models_portfolio: set[str],
                 initial_df: pd.DataFrame,
                 next_update_dt: datetime.datetime,
                 ):
        super().__init__(next_update_dt, adaptation_goals, models_portfolio)
        self.node_id = node_id
        self.base_model_metadata = base_model_metadata
        self.initial_df = initial_df

    def to_dict(self) -> dict[str, Any]:
        predictor_goals = [goal.to_dict() for goal in self.predictor_adaptation_goals]
        return {
            'node_id': self.node_id,
            'base_model_metadata': self.base_model_metadata.to_dict(),
            'predictor_adaptation_goals': predictor_goals,
            'models_portfolio': list(self.models_portfolio),
            'initial_df': self.initial_df.to_json() if self.initial_df is not None else None
        }
