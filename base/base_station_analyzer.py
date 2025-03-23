import datetime
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from base.base_station_knowledge import BaseStationKnowledge
from base.base_station_planner import BaseStationPlanner
from base.model import Model
from base.node_manager import NodeManager
from base.portfolio_analysis_result import PortfolioAnalysisResult
from common.data_storage import DataStorage
from common.predictor import Predictor


class BaseStationAnalyzer(ABC):

    def __init__(self, knowledge: BaseStationKnowledge, planner: BaseStationPlanner):
        self.knowledge = knowledge
        self.planner = planner

    @abstractmethod
    def analyze(self, dt: datetime.datetime, node: NodeManager) -> None:
        pass


class PredictorGoalsBaseStationAnalyzer(BaseStationAnalyzer):

    def __init__(self, knowledge: BaseStationKnowledge, planner: BaseStationPlanner):
        super().__init__(knowledge, planner)

    def analyze(self, dt: datetime.datetime, node: NodeManager) -> None:
        adaptation_goals = self.knowledge.portfolio_adaptation_goals

        # Adaptation required?
        if any(goal.requires_adaptation(dt, node) for goal in adaptation_goals):

            # Generate new model(s)
            logging.info(f"{dt.isoformat()} BS Analyzer: Generating new model portfolio.")
            candidate_models: set[Model] = self.knowledge.learning_strategy.get_candidate_models(node, dt)

            # At least one candidate model?
            if len(candidate_models) > 0:
                logging.info(f"{dt.isoformat()} BS Analyzer: Evaluating current models.")
                current_model_ids = node.get_model_ids()
                current_models: set[Model] = self.knowledge.model_manager.get_models(current_model_ids)
                baseline_results: list[PortfolioAnalysisResult] = self._evaluate_models(dt, node, current_models)

                logging.info(f"{dt.isoformat()} BS Analyzer: Evaluating candidate models.")
                candidate_results: list[PortfolioAnalysisResult] = self._evaluate_models(dt, node, candidate_models)
                self.planner.plan(dt, node, baseline_results, candidate_results)
            else:
                logging.info(f"{dt.isoformat()} BS Analyzer: No candidate model found. Use fail-safe strategy.")
                self.planner.fail_safe_plan(dt, node)

    def _evaluate_models(
            self, dt: datetime.datetime, node: NodeManager, models: set[Model]
    ) -> list[PortfolioAnalysisResult]:
        results: list[PortfolioAnalysisResult] = []
        logging.info(
            f"{dt.isoformat()} BS Analyzer: Analyzing models {[model.metadata.uuid for model in models]}."
        )
        for model in models:
            evaluation = self._evaluate_model(dt, model, node)
            results.append(evaluation)

        return results

    def _evaluate_model(
            self, dt: datetime.datetime, model: Model, node: NodeManager
    ) -> PortfolioAnalysisResult:
        """
        Evaluates a model candidate against a set of nodes to adapt and their goals.

        Args:
            dt: the datetime of the event that triggered the analysis
            model: the model to evaluate
            node: the node that requires adaptation

        Returns:
            the evaluation results of the model for each node
        """
        evaluation = {}
        data_reduction_strategy = self.knowledge.data_reduction_strategy
        data: DataStorage = node.predictor.data
        score = 0

        new_predictor = Predictor(model, data, data_reduction_strategy)
        goals = node.get_adaptation_goals()
        for goal in goals:
            score += goal.evaluate(new_predictor, dt) / len(goals)
        evaluation[node.node_id] = score
        logging.info(f"BS Analyzer: Model {model.metadata.uuid} score: {score}")
        return PortfolioAnalysisResult(model, pd.Series(dtype=np.float64, data=evaluation))

    def get_nodes_that_require_adaptation(self, dt: datetime.datetime) -> list[NodeManager]:
        """
        Assess the status of all nodes in the cluster, finding the nodes that require adaptation.

        Args:
            dt: the datetime of the triggering event.

        Returns:
            The list of nodes in the cluster that need adaptation.
        """
        goals = self.knowledge.portfolio_adaptation_goals
        nodes = self.knowledge.get_nodes().values()
        nodes_to_adapt: list[NodeManager] = []
        for node in nodes:
            if any(goal.requires_adaptation(dt, node) for goal in goals):
                logging.info(f"{dt.isoformat()} BS Analyzer: node {node.node_id} requires adaptation")
                nodes_to_adapt.append(node)
        return nodes_to_adapt
