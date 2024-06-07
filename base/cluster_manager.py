import os
import threading
from datetime import datetime
from typing import Optional

import pandas as pd

from base import Config
from base.model import Model, ModelID
from base.model_portfolio import ModelPortfolio
from base.model_trainer import DefaultModelTrainer
from base.node_manager import NodeManager, NodeID
from common import ThresholdMetric, ModelMetadata

ClusterID = str


class ClusterManager:

    def __init__(self, config: Config):
        self._nodes: dict[NodeID, NodeManager] = {}
        self._model_portfolio = ModelPortfolio(config)
        self._model_trainer = DefaultModelTrainer(epochs=2)
        self._training_df: pd.DataFrame = pd.read_pickle(config.training_data_pickle_path)
        self._training_threads: dict[NodeID, threading.Thread] = {}

    def add_node(self, node_id: NodeID, threshold_metric: ThresholdMetric, data: pd.DataFrame = None) -> None:
        model = self._model_portfolio.base_model
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

    def get_current_model(self, node_id: NodeID) -> Model:
        return self._get_node(node_id).model

    def get_models_in_portfolio(self) -> list[ModelID]:
        """
        :return: The list of ModelIDs of the models in the portfolio.
        """
        return self._model_portfolio.get_available_models()

    def get_model_upload_path(self, model_id: ModelID) -> os.path:
        return self._model_portfolio.get_model_tflite_file_path(model_id)

    def get_model_metadata(self, model_id: ModelID) -> ModelMetadata:
        return self._model_portfolio.get_model(model_id).metadata

    def handle_new_model_request(self, node_id: NodeID) -> Optional[Model]:
        """
        Handles a new model request sent by a Node.

        :param node_id: The ID of the node that requested a new model.
        """
        # TODO: implement adaptation logic to determine the right moment to train a new model
        node_manager = self._get_node(node_id)
        measurements = node_manager.get_measurements()
        metadata = node_manager.model.metadata
        if self._training_threads.get(node_id) is None or not self._training_threads[node_id].is_alive():
            training_thread = threading.Thread(target=self._train_new_model, args=(metadata, measurements))
            self._training_threads[node_id] = training_thread
            training_thread.start()
        return None

    def _train_new_model(self, model_metadata: ModelMetadata, data: pd.DataFrame = None) -> Model:
        """
        Trains a new model for the provided Node with the provided metadata, using the provided data as training set.
        The new model is then saved into the Cluster's portfolio.

        :param model_metadata: the metadata of the new model
        :param data: the training data
        :return: the new Model
        """
        base_model: Model = self._model_portfolio.base_model
        new_model = self._model_trainer.train_new_model(base_model.model, model_metadata, data)
        self._model_portfolio.save_model(new_model)
        return new_model

    def _get_data(self, node_id: NodeID, start: datetime, end: datetime) -> pd.DataFrame:
        node_manager = self._get_node(node_id)
        return node_manager.get_measurements_between(start, end)

    def _get_node(self, node_id: NodeID) -> NodeManager:
        node = self._nodes.get(node_id)
        if node is None:
            raise ValueError(f'Node id {node_id} is not valid.')
        return node
