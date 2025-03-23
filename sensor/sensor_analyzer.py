import datetime
import logging
from abc import ABC, abstractmethod

import pandas as pd
from numpy import float64

from common.data_storage import DataStorage
from common.prediction_model import PredictionModel
from common.predictor import Predictor
from sensor.predictor_configuration import PredictorConfiguration
from sensor.sensor_analysis_result import PredictorAnalysisResult
from sensor.sensor_knowledge import SensorKnowledge
from sensor.sensor_planner import SensorPlanner


class SensorAnalyzer(ABC):

    def __init__(self,
                 knowledge: SensorKnowledge,
                 planner: SensorPlanner
                 ):
        self.knowledge = knowledge
        self.planner = planner

    @abstractmethod
    def analyze(self, dt: datetime.datetime) -> None:
        pass


class PortfolioSensorAnalyzer(SensorAnalyzer):

    def __init__(self,
                 knowledge: SensorKnowledge,
                 planner: SensorPlanner
                 ):
        super().__init__(knowledge, planner)

    def analyze(self, dt: datetime.datetime) -> None:
        knowledge = self.knowledge
        current_predictor = knowledge.predictor
        adaptation_goals = knowledge.adaptation_goals

        if knowledge.pending_adaptation:
            logging.info(f"{dt.isoformat()} SN Analyzer: adaptation pending. Skipping analysis.")
            return

        # Assess adaptation goals
        if any(goal.requires_adaptation(current_predictor, dt) for goal in adaptation_goals):
            knowledge.pending_adaptation = True

            logging.info(f"{dt.isoformat()} SN Analyzer: evaluating models in portfolio.")
            analysis_results: list[PredictorAnalysisResult] = self._execute_analysis(dt)

            # Is there at least one suitable model?
            if len(analysis_results) > 0:
                self.planner.plan(dt, analysis_results)
            else:
                self.planner.fail_safe_plan(dt)

            knowledge.pending_adaptation = False
        else:
            logging.info(f"{dt.isoformat()} SN Analyzer: no adaptation required.")
            knowledge.update_prediction_horizon(dt)
            knowledge.adjust_predictions(
                dt,
                knowledge.predictor.get_measurement_at(dt),
                knowledge.predictor.get_prediction_at(dt)
            )

    def _update_portfolio_and_evaluate_new_models(self, dt: datetime.datetime) -> list[PredictorAnalysisResult]:
        knowledge = self.knowledge
        current_predictor = knowledge.predictor
        model_manager = knowledge.model_portfolio

        results: list[PredictorAnalysisResult] = []

        data = current_predictor.data
        new_model_ids = self.knowledge.synchronize_configurations(dt)

        logging.debug(f"{dt.isoformat()} SN Analyzer: Analyzing models {list(new_model_ids)}")
        for model_id in new_model_ids:
            model = model_manager.get_model(model_id)
            result = self._evaluate_model(data, dt, model)
            if not result.evaluation.empty and result.evaluation.sum() > 0:
                results.append(result)

        return results

    def _execute_analysis(self, dt: datetime.datetime) -> list[PredictorAnalysisResult]:
        current_predictor = self.knowledge.predictor
        models = self.knowledge.model_portfolio.get_models()
        data = current_predictor.data

        return self._evaluate_models(current_predictor, data, dt, models)

    def _evaluate_models(
            self,
            current_predictor: Predictor,
            data: DataStorage,
            dt: datetime.datetime,
            models_to_analyze: dict[str, PredictionModel]
    ) -> list[PredictorAnalysisResult]:
        analysis_results = []
        models = models_to_analyze.items()

        if len(models) < 2:
            return analysis_results

        logging.info(f"{dt.isoformat()} SN Analyzer: Analyzing models {list(models_to_analyze.keys())}")
        # Evaluate all available prediction models
        for model_id, model in models:
            result: PredictorAnalysisResult = self._evaluate_model(data, dt, model)
            current_predictor.data.add_analysis(dt, result.configuration.option_id, result.evaluation.to_json)

            # If the results are above the minimum threshold (i.e., at least one goal has score > 0 for this model),
            # create an analysis results item and add it to the list
            if not result.evaluation.empty and result.evaluation.sum() > 0:
                analysis_results.append(result)
        return analysis_results

    def _evaluate_model(
            self,
            data: DataStorage,
            dt: datetime.datetime,
            model: PredictionModel
    ) -> PredictorAnalysisResult:
        data_reduction_strategy = self.knowledge.data_reduction_strategy
        new_predictor = Predictor(model, data, data_reduction_strategy)
        model_id = model.metadata.uuid
        adaptation_goals = self.knowledge.adaptation_goals

        evaluation = {}  # Create a dict with every goal's score for the current configuration

        for goal in adaptation_goals:
            score = goal.evaluate(new_predictor, dt)
            if score > 0:
                evaluation[goal.goal_id] = score
        configuration = PredictorConfiguration(model_id, new_predictor)
        evaluation_series = pd.Series(dtype=float64, data=evaluation)
        logging.info(f"SN Analyzer: Model {model.metadata.uuid} score: {evaluation_series.sum()}")
        return PredictorAnalysisResult(configuration, evaluation_series)
