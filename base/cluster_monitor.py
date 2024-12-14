import datetime
import logging
from abc import ABC, abstractmethod

from base.cluster_analyzer import ClusterAnalyzer
from base.cluster_knowledge import ClusterKnowledge


class ClusterMonitor(ABC):

    def __init__(self, knowledge: ClusterKnowledge, analyzer: ClusterAnalyzer):
        self.knowledge = knowledge
        self.analyzer = analyzer

    @abstractmethod
    def monitor(self, dt: datetime.datetime, node_id: str) -> None:
        pass


class ViolationsClusterMonitor(ClusterMonitor):

    def __init__(self, knowledge: ClusterKnowledge, analyzer: ClusterAnalyzer):
        super().__init__(knowledge, analyzer)

    def monitor(self, dt: datetime.datetime, node_id: str) -> None:
        cluster_goals = self.knowledge.cluster_adaptation_goals
        node = self.knowledge.get_node(node_id)

        if node.last_adaptation_dt is None:
            node.last_adaptation_dt = dt

        # Check adaptation goals for violations
        if any(goal.is_violation(dt, node) for goal in cluster_goals):
            logging.info(f"{dt.isoformat()} BS Monitor: violation of cluster goals detected")
            self.analyzer.analyze(dt, node)
