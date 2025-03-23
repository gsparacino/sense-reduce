import datetime
import logging
from abc import ABC, abstractmethod

from base.base_station_analyzer import BaseStationAnalyzer
from base.base_station_knowledge import BaseStationKnowledge


class BaseStationMonitor(ABC):

    def __init__(self, knowledge: BaseStationKnowledge, analyzer: BaseStationAnalyzer):
        self.knowledge = knowledge
        self.analyzer = analyzer

    @abstractmethod
    def monitor(self, dt: datetime.datetime, node_id: str) -> None:
        pass


class ViolationsBaseStationMonitor(BaseStationMonitor):

    def __init__(self, knowledge: BaseStationKnowledge, analyzer: BaseStationAnalyzer):
        super().__init__(knowledge, analyzer)

    def monitor(self, dt: datetime.datetime, node_id: str) -> None:
        cluster_goals = self.knowledge.portfolio_adaptation_goals
        node = self.knowledge.get_node(node_id)

        if node.last_adaptation_dt is None:
            node.last_adaptation_dt = dt

        # Check adaptation goals for violations
        if any(goal.is_violation(dt, node) for goal in cluster_goals):
            logging.info(f"{dt.isoformat()} BS Monitor: violation of portfolio goals detected")
            self.analyzer.analyze(dt, node)
