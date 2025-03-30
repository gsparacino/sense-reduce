import datetime
import logging
from abc import ABC, abstractmethod
from typing import List, Callable, Optional

from sensor.predictor_configuration import PredictorConfiguration
from sensor.sensor_adaptation_actions import update_and_adjust_predictions, notify_violation
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
            adaptation_actions.append(update_and_adjust_predictions)
            self.executor.execute(dt, adaptation_actions)
        else:
            logging.info(f"{dt.isoformat()} SN Planner: unable to find a better model, falling back to fail-safe plan")
            self.fail_safe_plan(dt)

    def fail_safe_plan(self, dt: datetime.datetime):
        adaptation_actions: list[Callable[[datetime.datetime, SensorKnowledge], None]] = []

        logging.info(f"{dt.isoformat()} SN Planner: executing fail-safe plan")
        adaptation_actions.append(notify_violation)
        adaptation_actions.append(update_and_adjust_predictions)

        self.executor.execute(dt, adaptation_actions)

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
