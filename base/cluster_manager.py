import os
from datetime import datetime
from typing import List

import pandas as pd

from base import Config
from base.base_adaptive_strategy import BaseStationAdaptiveStrategy
from base.model import Model, ModelID
from base.model_manager import ModelManager
from base.model_trainer import ModelTrainer
from base.node_manager import NodeManager, NodeID
from common import ThresholdMetric, ModelMetadata


class ClusterManager:

    def __init__(self,
                 config: Config,
                 model_manager: ModelManager,
                 model_trainer: ModelTrainer,
                 adaptive_strategy: BaseStationAdaptiveStrategy
                 ):
        self._nodes: dict[NodeID, NodeManager] = {}
        self._model_manager = model_manager
        self._model_trainer = model_trainer
        self._training_df: pd.DataFrame = pd.read_pickle(config.training_data_pickle_path)
        # TODO: extension idea -> ClusterManager can change strategy at runtime, depending on the context
        self._adaptive_strategy = adaptive_strategy

    def add_node(self, node_id: NodeID, threshold_metric: ThresholdMetric, data: pd.DataFrame = None) -> None:
        """
        Adds a new node to the cluster handled by this ClusterManager.

        :param node_id: the id of the node to add
        :param threshold_metric: the threshold metric to use for the new node
        :param data: the measurements to preload on the new node
        """
        model = self._model_manager.base_model
        node_manager = NodeManager(node_id, threshold_metric, model)
        if data is None:
            node_manager.add_measurements(self._training_df)
        else:
            node_manager.add_measurements(data)
        self._nodes[node_id] = node_manager

    def add_measurements(self, node_id: NodeID, measurements: pd.DataFrame) -> None:
        """
        :param node_id: the id of a node
        :param measurements: the dataframe of measurements that will be added to the node's measurements
        """
        node = self._get_node(node_id)
        node.add_measurements(measurements)

    def get_measurements(self, node_id: NodeID) -> pd.DataFrame:
        """
        :param node_id: the id of a node
        :return: all the measurements persisted for the given node
        """
        return self._get_node(node_id).get_measurements()

    def add_violation(self, node_id: NodeID, timestamp: datetime, model_id: ModelID) -> None:
        """
        :param node_id: the id of the node that reported a violation
        :param timestamp: the datetime of the violation
        :param model_id: the model that caused the violation
        """
        self._get_node(node_id).add_violation(timestamp, model_id)

    def get_current_model(self, node_id: NodeID) -> Model:
        """
        :param node_id: the id of a node
        :return: the Model that is currently active on the given node
        """
        return self._get_node(node_id).model

    def set_current_model(self, node_id: NodeID, model_id: ModelID) -> None:
        """
        :param node_id: the id of a node
        :param model_id: the model that is currently active on the given node
        """
        model = self._model_manager.get_model(model_id)
        self._get_node(node_id).model = model

    def get_recommended_models(self, node_id: NodeID) -> list[ModelID]:
        """
        :param node_id: the id of a node
        :return: the list of recommended models for the given node
        """
        return self._adaptive_strategy.get_recommended_models(self._get_node(node_id), self._get_cluster_nodes())

    def get_model_upload_path(self, model_id: ModelID) -> os.path:
        """
        :param model_id: the id of a model
        :return: the os.path of the requested model, i.e. the file system path of the model's files
        """
        return self._model_manager.get_model_tflite_file_path(model_id)

    def get_model_metadata(self, model_id: ModelID) -> ModelMetadata:
        """
        :param model_id: the id of a model
        :return: the ModelMetadata of the requested model
        """
        return self._model_manager.get_model(model_id).metadata

    def handle_new_model_request(self, node_id: NodeID, node_portfolio: List[ModelID]) -> None:
        """
        Handles a new model request sent by a Node.

        :param node_portfolio: the node's current portfolio of models.
        :param node_id: The ID of the node that requested a new model.
        """
        self._adaptive_strategy.handle_new_model_request(
            self._get_node(node_id), node_portfolio, self._get_cluster_nodes()
        )

    def _get_data(self, node_id: NodeID, start: datetime, end: datetime) -> pd.DataFrame:
        node_manager = self._get_node(node_id)
        return node_manager.get_measurements_between(start, end)

    def _get_node(self, node_id: NodeID) -> NodeManager:
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f'Node id {node_id} is not valid.')
        return node

    def _get_cluster_nodes(self) -> List[NodeManager]:
        return list(self._nodes.values())
