import os

from base import Config
from base.model import Model, ModelID
from common.model_utils import clone_model, load_model_from_savemodel, save_model, get_model_tflite_path, \
    get_model_dir_path


class ModelPortfolio:
    def __init__(
            self,
            config: Config
    ):
        self._config = config
        self.base_model: Model = self._load_base_model(config)

    def clone_model(self, model: Model) -> Model:
        """
        Clones a model for the provided Node, using the provided model's metadata.

        :param model: the model to clone
        :return: the new model
        """
        model = clone_model(model)
        self.save_model(model)
        return model

    def load_model(self, model_id: ModelID) -> Model:
        """
        Loads the model with the provided ID for the provided Node
        '
        :param model_id: the ID of the model to load
        :return: the loaded model
        """
        model_path = os.path.join(self._config.model_dir, model_id)
        return load_model_from_savemodel(model_path)

    def get_model_tflite_file_path(self, model_id: ModelID) -> os.path:
        path = get_model_dir_path(self._config.model_dir, model_id)
        return get_model_tflite_path(path, model_id)

    def save_model(self, model: Model) -> None:
        """
        Saves a model as a file in the base station's model directory.

        :param model: the model to save
        """
        path = get_model_dir_path(self._config.model_dir, model.model_id)
        save_model(model, path)

    def _load_base_model(self, config: Config) -> Model:
        """
        Loads the cluster's default model.

        :param config: The Base Station's configuration
        :return: The base model
        """
        model_id = config.base_model_id
        return self.load_model(model_id)
