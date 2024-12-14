import datetime
import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import pandas as pd

from base.cluster_analysis_result import ClusterAnalysisResult
from base.cluster_knowledge import ClusterKnowledge
from base.cluster_planner import ClusterPlanner
from base.model import Model
from base.node_manager import NodeManager
from common.data_storage import DataStorage
from common.predictor import Predictor


class ClusterAnalyzer(ABC):

    def __init__(self, knowledge: ClusterKnowledge, planner: ClusterPlanner):
        self.knowledge = knowledge
        self.planner = planner

    @abstractmethod
    def analyze(self, dt: datetime.datetime, node: NodeManager) -> None:
        pass


class PortfolioClusterAnalyzer(ClusterAnalyzer):

    def __init__(self, knowledge: ClusterKnowledge, planner: ClusterPlanner):
        super().__init__(knowledge, planner)

    def analyze(self, dt: datetime.datetime, node: NodeManager) -> None:
        # Assess cluster status
        nodes_to_adapt: list[NodeManager] = self.get_nodes_that_require_adaptation(dt)

        # Adaptation required?
        if len(nodes_to_adapt) > 0:
            models: List[Model] = self.knowledge.model_manager.get_all_models()
            results: list[ClusterAnalysisResult] = []

            logging.info(f"{dt.isoformat()} BS Analyzer: Analyzing models {[model.metadata.uuid for model in models]}")
            for model in models:
                evaluation = self._evaluate_model(dt, model, nodes_to_adapt)
                if evaluation.get_score() > 0:
                    results.append(evaluation)

            # At least one suitable model?
            if len(results) > 0:
                self.planner.plan(dt, nodes_to_adapt, results)
            else:
                logging.info(f"{dt.isoformat()} BS Analyzer: No suitable model found. Switching to fail-safe plan.")
                self.planner.fail_safe_plan(dt, node)
        else:
            logging.info(f"{dt.isoformat()} BS Analyzer: none of the nodes require adaptation.")

    def _evaluate_model(
            self, dt: datetime.datetime, model: Model, nodes_to_adapt: list[NodeManager]
    ) -> ClusterAnalysisResult:
        """
        Evaluates a model candidate against a set of nodes to adapt and their goals.

        Args:
            dt: the datetime of the event that triggered the analysis
            model: the model to evaluate
            nodes_to_adapt: the list of nodes that require adaptation

        Returns:
            the evaluation results of the model for each node
        """
        evaluation = {}
        data_reduction_strategy = self.knowledge.data_reduction_strategy
        for node in nodes_to_adapt:
            score = 0
            if node.get_active_model_id() == model.metadata.uuid:
                evaluation[node.node_id] = score
                continue
            data: DataStorage = node.data
            new_predictor = Predictor(model, data, data_reduction_strategy)
            goals = node.get_adaptation_goals()
            for goal in goals:
                score += goal.evaluate(new_predictor, data_reduction_strategy, dt) / len(goals)
            evaluation[node.node_id] = score
        return ClusterAnalysisResult(model, pd.Series(dtype=np.float64, data=evaluation))

    def get_nodes_that_require_adaptation(self, dt: datetime.datetime) -> list[NodeManager]:
        """
        Assess the status of all nodes in the cluster, finding the nodes that require adaptation.

        Args:
            dt: the datetime of the triggering event.

        Returns:
            The list of nodes in the cluster that need adaptation.
        """
        goals = self.knowledge.cluster_adaptation_goals
        nodes = self.knowledge.get_nodes().values()
        nodes_to_adapt: list[NodeManager] = []
        for node in nodes:
            if any(goal.requires_adaptation(dt, node) for goal in goals):
                logging.info(f"{dt.isoformat()} BS Analyzer: node {node.node_id} requires adaptation")
                nodes_to_adapt.append(node)
        return nodes_to_adapt
