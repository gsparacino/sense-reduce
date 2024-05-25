import os

import pandas as pd

from base import Config
from base.model import Model, ModelID
from base.model_trainer import DefaultModelTrainer
from base.node_manager import NodeID
from common import ModelMetadata
from common.model_utils import clone_model, load_model_from_savemodel, save_model


class ModelManager:
    def __init__(
            self,
            config: Config
    ):
        self._model_trainer = DefaultModelTrainer(epochs=2)
        self._config = config
        self.base_model: Model = self._load_base_model(config)

    def train_new_model(self, node_id: NodeID, model_metadata: ModelMetadata, data: pd.DataFrame = None) -> Model:
        """
        Trains a new model for the provided Node with the provided metadata, using the provided data as training set.

        :param node_id: the ID of the node for which the model should be trained
        :param model_metadata: the metadata of the new model
        :param data: the training data
        :return: the new Model
        """
        new_model = self._model_trainer.train_new_model(self.base_model.model, self.base_model.metadata, data)
        self._save_model(node_id, new_model)
        return new_model

    def clone_model(self, node_id: NodeID, model: Model) -> Model:
        """
        Clones a model for the provided Node, using the provided model's metadata.

        :param node_id: the ID of the node for which the model should be created
        :param model: the model to clone
        :return: the new model
        """
        model = clone_model(model)
        self._save_model(node_id, model)
        return model

    def load_model(self, model_id: ModelID, node_id: NodeID = None) -> Model:
        """
        Loads the model with the provided ID for the provided Node
        '
        :param model_id: the ID of the model to load
        :param node_id: the ID of the node for which the model should be loaded
        :return: the loaded model
        """
        model_path = os.path.join(self._config.model_dir, node_id, model_id) if node_id \
            else os.path.join(self._config.model_dir, model_id)
        return load_model_from_savemodel(model_path)

    def get_model_file_path(self, model_id: ModelID, node_id: NodeID) -> os.path:
        return os.path.join(self._config.model_dir, node_id, model_id, f"{model_id}.tflite")

    def _save_model(self, node_id: NodeID, model: Model) -> None:
        """
        Saves a model as a file in the base station's model directory.

        :param node_id: the ID of the node to which the model is associated
        :param model: the model to save
        """
        model_path = os.path.join(self._config.model_dir, node_id, model.model_id)
        save_model(model, model_path)

    def _load_base_model(self, config: Config) -> Model:
        model_id = config.base_model_id
        return self.load_model(model_id)
