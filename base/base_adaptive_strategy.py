import logging
import threading
from abc import ABC, abstractmethod
from typing import List

import pandas as pd

from base import Config
from base.model import ModelID, Model
from base.model_manager import ModelManager
from base.learning_strategy import LearningStrategy
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


class FixedPortfolioAdaptiveStrategy(BaseStationAdaptiveStrategy):

    def __init__(self, model_manager: ModelManager):
        """
        An adaptive strategy that keeps the initial set of models, without training new ones. Requests for new models
        coming from SNs are ignored.

        Args:
            model_manager: a ModelManager that handles I/O of all models in the portfolio
        """
        self._model_manager = model_manager

    def get_recommended_models(self, node: NodeManager, cluster_nodes: List[NodeManager]) -> List[ModelID]:
        return self._model_manager.get_all_models()

    def handle_new_model_request(self, node: NodeManager, node_portfolio: List[ModelID],
                                 cluster_nodes: List[NodeManager]) -> None:
        pass


class TrainAndDeployAdaptiveStrategy(BaseStationAdaptiveStrategy):

    def __init__(self, config: Config, model_manager: ModelManager, learning_strategy: LearningStrategy):
        """
        An adaptive strategy that trains new models whenever an SN requests one.

        Args:
            config: a Config object with the BS configuration parameters
            model_manager: a ModelManager that handles I/O of all models in the portfolio
            learning_strategy: the LearningStrategy used to train new models when necessary
        """
        self._config = config
        self._model_manager = model_manager
        self._learning_strategy = learning_strategy
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
        new_model = self._learning_strategy.train_new_model(base_model.model, model_metadata, data)
        self._model_manager.save_model(new_model)
        return new_model


def adaptive_strategy_factory(learning_strategy: LearningStrategy,
                              model_manager: ModelManager,
                              config: Config) -> BaseStationAdaptiveStrategy:
    if config.adaptive_strategy:
        match config.adaptive_strategy.lower():
            case "train_deploy":
                logging.debug("Using TrainAndDeployAdaptiveStrategy as BaseStationAdaptiveStrategy")
                return TrainAndDeployAdaptiveStrategy(config, model_manager, learning_strategy)
            case "fixed":
                logging.debug("Using FixedPortfolioAdaptiveStrategy as BaseStationAdaptiveStrategy")
                return FixedPortfolioAdaptiveStrategy(model_manager)
            case _:
                logging.debug(
                    "No strategy matches config.yaml's parameter adaptive_strategy, using default "
                    "(FixedPortfolioAdaptiveStrategy)"
                )
                return FixedPortfolioAdaptiveStrategy(model_manager)
    logging.debug("Missing parameter adaptive_strategy in config.yaml, using default (FixedPortfolioAdaptiveStrategy)")
