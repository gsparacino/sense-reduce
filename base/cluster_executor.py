import datetime
from abc import ABC, abstractmethod
from typing import Callable

from base.cluster_knowledge import ClusterKnowledge


class ClusterExecutor(ABC):

    def __init__(self, knowledge: ClusterKnowledge):
        self.knowledge = knowledge

    @abstractmethod
    def execute(self,
                dt: datetime.datetime,
                adaptation_actions: list[Callable[[datetime.datetime, ClusterKnowledge], None]]
                ):
        pass


class SequentialClusterExecutor(ClusterExecutor):

    def __init__(self, knowledge: ClusterKnowledge):
        super().__init__(knowledge)

    def execute(self,
                dt: datetime.datetime,
                adaptation_actions: list[Callable[[datetime.datetime, ClusterKnowledge], None]]
                ):
        for action in adaptation_actions:
            action(dt, self.knowledge)
