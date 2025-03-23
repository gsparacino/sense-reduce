import datetime
import logging
from typing import Type

import pandas as pd

from common.data_reduction_strategy import DataReductionStrategy
from sensor.abstract_sensor import AbstractSensor
from sensor.base_station_adapter import BaseStationAdapter
from sensor.base_station_gateway import BaseStationGateway
from sensor.sensor_analyzer import SensorAnalyzer
from sensor.sensor_executor import SensorExecutor
from sensor.sensor_knowledge import SensorKnowledge
from sensor.sensor_monitor import SensorMonitor
from sensor.sensor_planner import SensorPlanner

# TODO make logging level configurable
logging.basicConfig(level=logging.INFO)


class SensorManager:

    def __init__(self,
                 model_dir: str,
                 input_features: list[str],
                 output_features: list[str],
                 sensor: AbstractSensor,
                 initial_df: pd.DataFrame,
                 base_station_gateway: BaseStationGateway,
                 monitor_type: Type[SensorMonitor],
                 analyzer_type: Type[SensorAnalyzer],
                 planner_type: Type[SensorPlanner],
                 executor_type: Type[SensorExecutor],
                 data_reduction_strategy: DataReductionStrategy,
                 sensor_knowledge: SensorKnowledge = None,
                 ):
        self.input_features = input_features
        self.output_features = output_features
        self.sensor = sensor
        self.data_reduction_strategy = data_reduction_strategy
        self.base_station = BaseStationAdapter(base_station_gateway, data_reduction_strategy)
        self.initial_df = initial_df
        self.knowledge = sensor_knowledge
        self.model_dir = model_dir
        self.monitor_type = monitor_type
        self.analyzer_type = analyzer_type
        self.planner_type = planner_type
        self.executor_type = executor_type

    # TODO improve MAPE models' configuration

    def run(self) -> None:
        if self.knowledge is None:
            knowledge_initialization = self.base_station.register_node(
                self.input_features,
                self.output_features
            )
            self.knowledge = SensorKnowledge.from_initialization(
                self.base_station, self.initial_df, knowledge_initialization, self.data_reduction_strategy,
                self.model_dir
            )
        executor = self.executor_type(self.knowledge)
        planner = self.planner_type(self.knowledge, executor)
        analyzer = self.analyzer_type(self.knowledge, planner)
        monitor = self.monitor_type(self.knowledge, analyzer)

        logging.info(f"{datetime.datetime.now().isoformat()} - Starting Sensor {self.knowledge.node_id}")

        while self.sensor.is_ready():
            monitor.monitor(self.sensor)
