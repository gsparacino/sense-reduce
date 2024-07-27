import logging
import threading
from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from base import Config
from base.model import ModelID, Model
from base.model_manager import ModelManager
from base.model_trainer import LearningStrategy
from base.node_manager import NodeManager, NodeID
from common import ModelMetadata
from common.resource_profiler import profiled


class BaseStationAdaptiveStrategy(ABC):

    @abstractmethod
    def get_recommended_models(self, node: NodeManager, cluster_nodes: List[NodeManager]) -> List[ModelID]:
        """
        :param node: the node to get recommended models for
        :param cluster_nodes: the nodes in the cluster of the given node
        :return: the list of recommended model IDs for the given node
        """
        pass

    @abstractmethod
    def handle_new_model_request(self,
                                 node: NodeManager, node_portfolio: List[ModelID], cluster_nodes: List[NodeManager]
                                 ) -> None:
        """
        Handles a new model request sent by a Node.

        :param node: the node that requested a new model
        :param node_portfolio: the node's current portfolio of models.
        :param cluster_nodes: the nodes currently in the cluster
        """
        pass


class DefaultBaseStationAdaptiveStrategy(BaseStationAdaptiveStrategy):

    def __init__(self, config: Config, model_manager: ModelManager, model_trainer: LearningStrategy):
        self._config = config
        self._model_manager = model_manager
        self._model_trainer = model_trainer
        self._training_threads: dict[NodeID, threading.Thread] = {}

    def get_recommended_models(self, node: NodeManager, cluster_nodes: List[NodeManager]) -> List[ModelID]:
        """
        :param node: the node to get recommended models for
        :param cluster_nodes: the nodes in the cluster of the given node
        :return: the list of recommended model IDs for the given node
        """
        return self._model_manager.get_all_models()

    def handle_new_model_request(self,
                                 node: NodeManager, node_portfolio: List[ModelID], cluster_nodes: List[NodeManager]
                                 ) -> None:
        """
        Handles a new model request sent by a Node.

        :param node: the node that requested a new model
        :param node_portfolio: the node's current portfolio of models.
        :param cluster_nodes: the nodes currently in the cluster
        """
        for recommended_model in self.get_recommended_models(node, node_portfolio):
            if recommended_model not in node_portfolio:
                # Node's Portfolio is not up-to-date with all the BS recommendations, do not train a new model
                # (let the node try the new recommended models first)
                logging.debug(
                    "Node's Portfolio is not up-to-date with all the BS recommendations, do not train a new model"
                )
                return

        measurements = node.get_measurements()
        metadata = node.model.metadata
        node_id = node.node_id
        if self._training_threads.get(node_id) is None or not self._training_threads[node_id].is_alive():
            training_thread = threading.Thread(target=self._train_new_model, args=(metadata, measurements))
            self._training_threads[node_id] = training_thread
            training_thread.start()

    @profiled(tag="Train new model")
    def _train_new_model(self, model_metadata: ModelMetadata, data: pd.DataFrame = None) -> Model:
        """
        Trains a new model for the provided Node with the provided metadata, using the provided data as training set.
        The new model is then saved into the Cluster's portfolio.

        :param model_metadata: the metadata of the new model
        :param data: the training data
        :return: the new Model
        """
        base_model: Model = self._model_manager.base_model
        new_model = self._model_trainer.train_new_model(base_model.model, model_metadata, data)
        self._model_manager.save_model(new_model)
        return new_model
