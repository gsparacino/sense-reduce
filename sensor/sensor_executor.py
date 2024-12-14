import datetime
from abc import ABC, abstractmethod
from typing import Callable

from sensor.sensor_knowledge import SensorKnowledge


class SensorExecutor(ABC):

    def __init__(self, knowledge: SensorKnowledge):
        self.knowledge = knowledge

    @abstractmethod
    def execute(self, dt: datetime.datetime,
                adaptation_actions: list[Callable[[datetime.datetime, SensorKnowledge], None]]
                ):
        pass


class SequentialSensorExecutor(SensorExecutor):

    def __init__(self, knowledge: SensorKnowledge):
        super().__init__(knowledge)

    def execute(self, dt: datetime.datetime,
                adaptation_actions: list[Callable[[datetime.datetime, SensorKnowledge], None]]
                ):
        for action in adaptation_actions:
            action(dt, self.knowledge)
