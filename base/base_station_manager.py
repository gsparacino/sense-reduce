import datetime
import logging
import uuid

import pandas as pd

from base.base_station_knowledge import BaseStationKnowledge
from base.base_station_monitor import BaseStationMonitor
from common.sensor_knowledge_update import SensorKnowledgeInitialization, SensorKnowledgeUpdate

logging.basicConfig(level=logging.INFO)


class BaseStationManager:

    def __init__(
            self,
            cluster_id: str,
            knowledge: BaseStationKnowledge,
            cluster_monitor: BaseStationMonitor,
    ):
        self.cluster_id = cluster_id
        self.knowledge = knowledge
        self.monitor = cluster_monitor

    def register_node(
            self,
            input_features: list[str],
            output_features: list[str],
            initial_df: pd.DataFrame,
    ) -> SensorKnowledgeInitialization:
        node_id = str(uuid.uuid4())
        logging.info(f"BS: Registering node {node_id}")
        data_reduction_strategy = self.knowledge.data_reduction_strategy
        return self.knowledge.add_node(node_id, input_features, output_features, initial_df, data_reduction_strategy)

    def handle_violation(
            self,
            node_id: str,
            dt: datetime.datetime,
            measurement: pd.Series,
            data: pd.DataFrame,
            configuration_id: str
    ) -> SensorKnowledgeUpdate:
        logging.info(f"{dt.isoformat()} BS: Node {node_id} notified a violation")
        node = self.knowledge.get_node(node_id)
        if node.get_active_model_id() != configuration_id:
            model = self.knowledge.model_manager.get_model(configuration_id)
            node.set_active_model(model, dt)
        node.predictor.add_measurement_df(data)
        node.predictor.add_violation(dt)
        node.predictor.update_prediction_horizon(dt)
        prediction = node.predictor.get_prediction_at(dt)
        node.predictor.adjust_to_measurement(dt, measurement.to_numpy(), prediction.to_numpy())
        self.monitor.monitor(dt, node_id)
        return node.to_sensor_knowledge_update()

    def sync(self,
             node_id: str
             ) -> SensorKnowledgeUpdate:
        node = self.knowledge.get_node(node_id)
        return node.to_sensor_knowledge_update()

    def handle_horizon_update(
            self,
            node_id: str,
            dt: datetime.datetime,
            data: pd.DataFrame,
            configuration_id: str
    ) -> SensorKnowledgeUpdate:
        logging.info(f"{dt.isoformat()} BS: Node {node_id} notified an horizon update")
        node = self.knowledge.get_node(node_id)
        if node.get_active_model_id() != configuration_id:
            model = self.knowledge.model_manager.get_model(configuration_id)
            node.set_active_model(model, dt)
        node.predictor.add_measurement_df(data)
        node.predictor.update_prediction_horizon(dt)
        self.monitor.monitor(dt, node_id)
        return node.to_sensor_knowledge_update()

    def add_measurement(
            self,
            node_id: str,
            dt: datetime.datetime,
            measurement: pd.Series
    ) -> SensorKnowledgeUpdate:
        logging.info(f"{dt.isoformat()} BS: Node {node_id} sent a measurement")
        node = self.knowledge.get_node(node_id)
        node.predictor.add_measurement(dt, measurement.to_numpy())
        self.monitor.monitor(dt, node_id)
        return node.to_sensor_knowledge_update()
