import datetime
from abc import ABC, abstractmethod
from typing import Callable

from base.base_station_knowledge import BaseStationKnowledge


class BaseStationExecutor(ABC):

    def __init__(self, knowledge: BaseStationKnowledge):
        self.knowledge = knowledge

    @abstractmethod
    def execute(self,
                dt: datetime.datetime,
                adaptation_actions: list[Callable[[datetime.datetime, BaseStationKnowledge], None]]
                ):
        pass


class SequentialBaseStationExecutor(BaseStationExecutor):

    def __init__(self, knowledge: BaseStationKnowledge):
        super().__init__(knowledge)

    def execute(self,
                dt: datetime.datetime,
                adaptation_actions: list[Callable[[datetime.datetime, BaseStationKnowledge], None]]
                ):
        for action in adaptation_actions:
            action(dt, self.knowledge)
