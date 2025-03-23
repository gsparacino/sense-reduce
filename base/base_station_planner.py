import datetime
import logging
from abc import ABC, abstractmethod
from typing import List, Callable, Optional

from base.base_station_configuration import BaseStationConfiguration
from base.base_station_executor import BaseStationExecutor
from base.base_station_knowledge import BaseStationKnowledge
from base.model import Model
from base.node_manager import NodeManager
from base.portfolio_analysis_result import PortfolioAnalysisResult


def _update_node_configuration(
        node_id: str,
        configuration: BaseStationConfiguration
) -> Callable[[datetime.datetime, BaseStationKnowledge], None]:
    def f(
            dt: datetime.datetime,
            knowledge: BaseStationKnowledge
    ) -> None:
        logging.info(
            f"{dt.isoformat()} BS Executor: updating {node_id} models: {[model.metadata.uuid for model in configuration.models]}")
        node_manager = knowledge.get_node(node_id)
        configuration.apply(dt, node_manager)
        node_manager.predictor.add_configuration_update(dt, configuration.option_id)

    return f


def _update_last_adaptation_dt(
        node_id: str,
):
    def f(
            dt: datetime.datetime,
            knowledge: BaseStationKnowledge
    ) -> None:
        logging.info(f"{dt.isoformat()} BS Executor: updating {node_id} last_adaptation_dt")
        node_manager = knowledge.get_node(node_id)
        node_manager.last_adaptation_dt = dt
        node_manager.next_sync_dt = node_manager.next_sync_dt

    return f


class BaseStationPlanner(ABC):

    def __init__(self, knowledge: BaseStationKnowledge, executor: BaseStationExecutor):
        self.knowledge = knowledge
        self.executor = executor

    @abstractmethod
    def plan(
            self,
            dt: datetime.datetime,
            node: NodeManager,
            baseline_results: List[PortfolioAnalysisResult],
            candidate_results: List[PortfolioAnalysisResult]
    ):
        pass

    @abstractmethod
    def fail_safe_plan(self, dt: datetime.datetime, node: NodeManager):
        pass


class PortfolioBaseStationPlanner(BaseStationPlanner):

    def __init__(self, knowledge: BaseStationKnowledge, executor: BaseStationExecutor):
        super().__init__(knowledge, executor)

    def plan(
            self,
            dt: datetime.datetime,
            node: NodeManager,
            baseline_results: List[PortfolioAnalysisResult],
            candidate_results: List[PortfolioAnalysisResult]
    ):
        adaptation_actions: list[Callable[[datetime.datetime, BaseStationKnowledge], None]] = []

        best_configuration = self._get_best_configuration(dt, node, baseline_results, candidate_results)
        if best_configuration is not None and not best_configuration.is_active(node):
            logging.info(f"{dt.isoformat()} BS Planner: composing adaptation plan for node {node.node_id}")
            adaptation_actions.append(_update_node_configuration(node.node_id, best_configuration))
            self.executor.execute(dt, adaptation_actions)
        else:
            self.fail_safe_plan(dt, node)

    def fail_safe_plan(self, dt: datetime.datetime, node: NodeManager):
        adaptation_actions: list[Callable[[datetime.datetime, BaseStationKnowledge], None]] = []

        logging.info(f"{dt.isoformat()} BS Planner: composing fail-safe plan for node {node.node_id}")
        adaptation_actions.append(_update_last_adaptation_dt(node.node_id))
        self.executor.execute(dt, adaptation_actions)

    def _get_next_sync_dt(self, dt: datetime.datetime, node: NodeManager):
        goals = self.knowledge.portfolio_adaptation_goals
        next_sync_dt = min(
            (
                goal.get_next_update_dt(dt, node) for goal in goals if
                goal.get_next_update_dt(dt, node) is not None
            ),
            default=None
        )
        return next_sync_dt

    def _get_best_configuration(
            self,
            dt: datetime.datetime,
            node: NodeManager,
            baseline_results: List[PortfolioAnalysisResult],
            candidate_results: List[PortfolioAnalysisResult]
    ) -> Optional[BaseStationConfiguration]:

        if len(candidate_results) > 0:

            # Check whether the new portfolio has a model that performs better than the baseline
            baseline_score: float = max([r.get_score() for r in baseline_results])
            candidate_score: float = max([r.get_score() for r in candidate_results])

            if self.knowledge.legacy_mode or candidate_score > baseline_score:
                models: set[Model] = set([r.model for r in candidate_results])
                next_sync_dt = self._get_next_sync_dt(dt, node)
                return BaseStationConfiguration(dt.isoformat(), models, next_sync_dt)

        return None
