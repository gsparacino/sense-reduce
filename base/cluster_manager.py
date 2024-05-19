import os
from datetime import datetime
from typing import Optional

import pandas as pd

from base import Config
from base.model import Model, ModelID
from base.model_manager import ModelManager
from base.node_manager import NodeManager, NodeID
from common import ThresholdMetric

ClusterID = str


class ClusterManager:

    def __init__(self, config: Config):
        self._nodes: dict[NodeID, NodeManager] = {}
        self.config: Config = config
        self.model_manager = ModelManager(config)
        self._training_df: pd.DataFrame = pd.read_pickle(config.training_data_pickle_path)

    def add_node(self, node_id: NodeID, threshold_metric: ThresholdMetric, data: pd.DataFrame = None) -> None:
        model = self.model_manager.clone_model(node_id, self.model_manager.base_model)
        node_manager = NodeManager(node_id, threshold_metric, model)
        if data is None:
            node_manager.add_measurements(self._training_df)
        else:
            node_manager.add_measurements(data)
        self._nodes[node_id] = node_manager

    def add_measurements(self, node_id: NodeID, measurements: pd.DataFrame) -> None:
        node = self._get_node(node_id)
        node.add_measurements(measurements)

    def get_measurements(self, node_id: NodeID) -> pd.DataFrame:
        return self._get_node(node_id).get_measurements()

    def get_current_model_of_node(self, node_id: NodeID) -> Model:
        return self._get_node(node_id).model

    def get_model_file_path(self, node_id: NodeID, model_id: ModelID) -> os.path:
        return self.model_manager.get_model_file_path(model_id, node_id)

    def handle_new_model_request(self, node_id: NodeID) -> Optional[Model]:
        node_manager = self._get_node(node_id)
        measurements = node_manager.get_measurements()
        metadata = node_manager.model.metadata
        new_model = self.model_manager.train_new_model(node_id, metadata, measurements)
        return new_model

    def _get_data(self, node_id: NodeID, start: datetime, end: datetime) -> pd.DataFrame:
        node_manager = self._get_node(node_id)
        return node_manager.get_measurements_between(start, end)

    def _get_node(self, node_id: NodeID) -> NodeManager:
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f'Node id {node_id} is not valid.')
        return node
