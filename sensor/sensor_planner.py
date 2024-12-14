import datetime
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Callable

import pandas as pd

from sensor.predictor_configuration import PredictorConfiguration
from sensor.sensor_analysis_result import PredictorAnalysisResult
from sensor.sensor_executor import SensorExecutor
from sensor.sensor_knowledge import SensorKnowledge


class SensorPlanner(ABC):

    def __init__(self, knowledge: SensorKnowledge, executor: SensorExecutor):
        self.knowledge = knowledge
        self.executor = executor

    @abstractmethod
    def plan(self, dt: datetime.datetime, analysis_results: List[PredictorAnalysisResult]):
        pass

    @abstractmethod
    def fail_safe_plan(self, dt: datetime.datetime):
        pass

    def _adjust_predictions_after_violation(self, dt: datetime.datetime, knowledge: SensorKnowledge) -> None:
        logging.info(f"{dt.isoformat()} SN Executor: adjusting predictions after violation")
        if knowledge.predictor.model_metadata.uuid not in knowledge.model_manager.get_model_ids():
            self.replace_deleted_model(dt, knowledge)
        else:
            self._refresh_prediction_horizon(dt)
        prediction: pd.Series = knowledge.predictor.get_prediction_at(dt)
        measurement: pd.Series = knowledge.predictor.data.get_measurements().loc[dt]
        self._adjust_prediction_horizon(dt, measurement, prediction)

    def _adjust_prediction_horizon(self, dt: datetime.datetime, measurement: pd.Series, prediction: pd.Series):
        self.knowledge.predictor.adjust_to_measurement(dt, measurement.to_numpy(), prediction.to_numpy())

    def _refresh_prediction_horizon(self, dt: datetime):
        logging.info(f"{dt.isoformat()} SN Executor: refreshing prediction horizon after violation")
        predictor = self.knowledge.predictor
        predictor.update_prediction_horizon(dt)

    def replace_deleted_model(self, dt: datetime.datetime, knowledge: SensorKnowledge):
        logging.info(f"{dt.isoformat()} SN Executor:  current model no longer in portfolio, replacing")
        new_model = next(iter(knowledge.model_manager.get_models().values()))
        knowledge.predictor.set_model(new_model, dt)
        knowledge.predictor.data.add_configuration_update(dt, new_model.metadata.uuid)

    def _handle_violation(self, dt: datetime.datetime, knowledge: SensorKnowledge) -> None:
        logging.info(f"{dt.isoformat()} SN Executor: notify violation to BS")
        predictor = knowledge.predictor
        measurement = knowledge.predictor.data.get_measurements().loc[dt]
        update = knowledge.base_station.on_violation(dt, predictor, measurement)
        if update is not None:
            knowledge.update(dt, update)

    def _find_best_configuration(
            self, analysis_results: List[PredictorAnalysisResult]
    ) -> Optional[PredictorConfiguration]:
        goals = self.knowledge.adaptation_goals
        best_configuration: Optional[PredictorConfiguration] = None
        scores = {}
        best_score = 0
        for analysis in analysis_results:
            score = 0
            for goal in goals:
                score += round(analysis.evaluation[goal.goal_id], 2)
            option_id = analysis.configuration.option_id
            scores[option_id] = score
            if score > best_score:
                best_score = score
                best_configuration = analysis.configuration

        if all([score == best_score for score in scores]):
            logging.info(
                f"SN Planner: all configurations have the same score, keeping current configuration")
            best_configuration = None

        return best_configuration


class PortfolioSensorPlanner(SensorPlanner):

    def __init__(self,
                 knowledge: SensorKnowledge,
                 executor: SensorExecutor
                 ):
        super().__init__(knowledge, executor)

    def plan(self, dt: datetime.datetime, analysis_results: List[PredictorAnalysisResult]):
        adaptation_actions: list[Callable[[datetime.datetime, SensorKnowledge], None]] = []

        if len(analysis_results) > 1:
            best_configuration = self._find_best_configuration(analysis_results)
        else:
            best_configuration = analysis_results[0].configuration

        if best_configuration is not None and not best_configuration.is_active(self.knowledge):
            logging.info(
                f"{dt.isoformat()} SN Planner: scheduling configuration update to {best_configuration.option_id}")
            adaptation_actions.append(best_configuration.apply)
            adaptation_actions.append(self._adjust_predictions_after_violation)
        else:
            logging.info(f"{dt.isoformat()} SN Planner: unable to find a better model, falling back to fail-safe plan")
            adaptation_actions = self._compose_fail_safe_plan(dt)

        self.executor.execute(dt, adaptation_actions)

    def fail_safe_plan(self, dt: datetime.datetime):
        logging.info(f"{dt.isoformat()} SN Planner: using fail-safe plan")
        adaptation_actions = self._compose_fail_safe_plan(dt)

        self.executor.execute(dt, adaptation_actions)

    def _compose_fail_safe_plan(self, dt: datetime.datetime):
        adaptation_actions: list[Callable[[datetime.datetime, SensorKnowledge], None]] = []
        if self.knowledge.synchronization_enabled(dt):
            logging.info(f"{dt.isoformat()} SN Planner: synchronization enabled, will send notification to BS")
            adaptation_actions.append(self._handle_violation)
        else:
            logging.info(
                f"{dt.isoformat()} SN Planner: synchronization disabled, will not send notification to BS"
            )
        adaptation_actions.append(self._adjust_predictions_after_violation)
        return adaptation_actions
