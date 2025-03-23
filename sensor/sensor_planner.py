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

    def _update_and_adjust_predictions(self, dt: datetime.datetime, knowledge: SensorKnowledge) -> None:
        if knowledge.predictor.model_metadata.uuid not in knowledge.model_portfolio.get_model_ids():
            self.replace_current_model(dt, knowledge)
        else:
            knowledge.update_prediction_horizon(dt)
        prediction: pd.Series = knowledge.predictor.get_prediction_at(dt)
        measurement: pd.Series = knowledge.predictor.data.get_measurements().loc[dt]
        knowledge.adjust_predictions(dt, measurement, prediction)

    def replace_current_model(self, dt: datetime.datetime, knowledge: SensorKnowledge):
        new_model = next(iter(knowledge.model_portfolio.get_models().values()))
        logging.info(
            f"{dt.isoformat()} SN Executor:  current model no longer in portfolio, replacing with {new_model.metadata.uuid}"
        )
        knowledge.predictor.set_model(new_model, dt)
        knowledge.predictor.data.add_configuration_update(dt, new_model.metadata.uuid)

    def _notify_violation(self, dt: datetime.datetime, knowledge: SensorKnowledge) -> None:
        logging.info(f"{dt.isoformat()} SN Executor: notify violation to BS")
        predictor = knowledge.predictor
        measurement = knowledge.predictor.data.get_measurements().loc[dt]
        update = knowledge.base_station.notify_violation(dt, predictor, measurement)
        if update is not None:
            new_portfolio: set[str] = update.models_portfolio
            logging.info(f"{dt.isoformat()} SN Executor: new portfolio received {[model for model in new_portfolio]}")
            knowledge.update(dt, update)
            if knowledge.predictor.model_metadata.uuid not in new_portfolio:
                self.replace_current_model(dt, knowledge)

    def _find_best_configuration(
            self, analysis_results: List[PredictorAnalysisResult]
    ) -> Optional[PredictorConfiguration]:

        if len(analysis_results) < 2:
            return analysis_results[0].configuration

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

        best_configuration = self._find_best_configuration(analysis_results)

        if best_configuration is not None and not best_configuration.is_active(self.knowledge):
            logging.info(
                f"{dt.isoformat()} SN Planner: scheduling configuration update to {best_configuration.option_id}"
            )
            adaptation_actions.append(best_configuration.apply)
            adaptation_actions.append(self._update_and_adjust_predictions)
            self.executor.execute(dt, adaptation_actions)
        else:
            logging.info(f"{dt.isoformat()} SN Planner: unable to find a better model, falling back to fail-safe plan")
            self.fail_safe_plan(dt)

    def fail_safe_plan(self, dt: datetime.datetime):
        adaptation_actions: list[Callable[[datetime.datetime, SensorKnowledge], None]] = []

        logging.info(f"{dt.isoformat()} SN Planner: executing fail-safe plan")
        adaptation_actions.append(self._notify_violation)
        adaptation_actions.append(self._update_and_adjust_predictions)

        self.executor.execute(dt, adaptation_actions)
