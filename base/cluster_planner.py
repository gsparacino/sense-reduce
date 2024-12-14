import datetime
import logging
from abc import ABC, abstractmethod
from typing import List, Callable, Optional

from base.cluster_analysis_result import ClusterAnalysisResult
from base.cluster_configuration import ClusterConfiguration
from base.cluster_executor import ClusterExecutor
from base.cluster_knowledge import ClusterKnowledge
from base.node_manager import NodeManager


def _update_node_configuration(
        node_id: str,
        configuration: ClusterConfiguration
) -> Callable[[datetime.datetime, ClusterKnowledge], None]:
    def f(
            dt: datetime.datetime,
            knowledge: ClusterKnowledge
    ) -> None:
        logging.info(f"Updating {node_id} models: {[model.metadata.uuid for model in configuration.models]}")
        node_manager = knowledge.get_node(node_id)
        configuration.apply(dt, node_manager)
        node_manager.data.add_configuration_update(dt, configuration.option_id)

    return f


class ClusterPlanner(ABC):

    def __init__(self, knowledge: ClusterKnowledge, executor: ClusterExecutor):
        self.knowledge = knowledge
        self.executor = executor

    @abstractmethod
    def plan(self, dt: datetime.datetime, nodes: list[NodeManager], analysis_results: List[ClusterAnalysisResult]):
        pass

    @abstractmethod
    def fail_safe_plan(self, dt: datetime.datetime, node: NodeManager):
        pass


class PortfolioClusterPlanner(ClusterPlanner):

    def __init__(self, knowledge: ClusterKnowledge, executor: ClusterExecutor):
        super().__init__(knowledge, executor)

    def plan(self, dt: datetime.datetime, nodes: list[NodeManager], analysis_results: List[ClusterAnalysisResult]):
        adaptation_actions: list[Callable[[datetime.datetime, ClusterKnowledge], None]] = []

        if len(analysis_results) == 0:
            return

        for node in nodes:
            best_configuration = self._get_best_configuration(dt, analysis_results)
            if best_configuration is not None and not best_configuration.is_active(node):
                logging.info(f"BS Planner: planning cluster configuration update to {best_configuration.option_id}")
                adaptation_actions.append(_update_node_configuration(node.node_id, best_configuration))

            if len(adaptation_actions) > 0:
                self.executor.execute(dt, adaptation_actions)

    def fail_safe_plan(self, dt: datetime.datetime, node: NodeManager):
        adaptation_actions: list[Callable[[datetime.datetime, ClusterKnowledge], None]] = []
        models = self.knowledge.learning_strategy.update_portfolio(node, dt)
        next_adaptation_dt = self._get_next_adaptation_dt(dt, node)
        configuration = ClusterConfiguration(dt.isoformat(), models, next_adaptation_dt)

        adaptation_actions.append(_update_node_configuration(node.node_id, configuration))

        self.executor.execute(dt, adaptation_actions)

    def _get_next_adaptation_dt(self, dt: datetime.datetime, node: NodeManager):
        goals = self.knowledge.cluster_adaptation_goals
        next_adaptation_dt = min(
            (
                goal.get_next_adaptation_dt(dt, node) for goal in goals if
                goal.get_next_adaptation_dt(dt, node) is not None
            ),
            default=None
        )
        return next_adaptation_dt

    def _get_best_configuration(
            self, dt: datetime.datetime, analysis_results: List[ClusterAnalysisResult]
    ) -> Optional[ClusterConfiguration]:
        node: NodeManager = next(iter(self.knowledge.get_nodes().values()))
        portfolio_size = len(node.get_model_ids())
        sorted_results = sorted(analysis_results, key=lambda r: r.get_score(), reverse=True)
        best_models = set([result.model for result in sorted_results[:portfolio_size]])
        next_adaptation_dt = self._get_next_adaptation_dt(dt, node)
        return ClusterConfiguration(dt.isoformat(), best_models, next_adaptation_dt)
