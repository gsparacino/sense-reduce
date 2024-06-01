import os
from typing import Dict, List

from base import Config
from base.model import Model, ModelID
from common import ModelMetadata
from common.model_utils import load_model_from_savemodel, save_model, get_model_tflite_path, \
    get_model_dir_path


class ModelPortfolio:

    def __init__(
            self,
            config: Config
    ):
        self._model_dir = config.model_dir
        self._models: Dict[ModelID, Model] = {}
        self._load_local_models()
        self.base_model: Model = self.get_model(config.base_model_id)

    # def clone_model(self, model: Model) -> Model:
    #     """
    #     Clones a model for the provided Node, using the provided model's metadata.
    #
    #     :param model: the model to clone
    #     :return: the new model
    #     """
    #     model = clone_model(model)
    #     self.save_model(model)
    #     return model
    def get_available_models(self) -> List[ModelID]:
        models: List[ModelMetadata] = list(model.model_id for _, model in self._models.items())
        return models

    def get_model(self, model_id: ModelID) -> Model:
        """
        Returns the model with the provided ModelID

        :param model_id: The ID of the model

        :return: The requested model

        :raises ValueError: if the provided ModelID does not match any model
        """
        model = self._models.get(model_id)
        if model is None:
            model = self._load_model(model_id)
            if model is None:
                raise ValueError(f'Model {model_id} not found')
            self._models[model_id] = model
        return model

    def _load_model(self, model_id: ModelID) -> Model:
        model_path = os.path.join(self._model_dir, model_id)
        model = load_model_from_savemodel(model_path)
        return model

    def get_model_tflite_file_path(self, model_id: ModelID) -> os.path:
        path = get_model_dir_path(self._model_dir, model_id)
        return get_model_tflite_path(path, model_id)

    def save_model(self, model: Model) -> Model:
        """
        Saves a model as a file in the base station's model directory.

        :param model: the model to save
        """
        model_id = model.model_id
        path = get_model_dir_path(self._model_dir, model_id)
        save_model(model, path)
        return self.get_model(model_id)

    def _load_local_models(self) -> None:
        """
        Initializes the Sensor's models portfolio by loading all TFLite models in its storage.
        """
        for item in os.scandir(self._model_dir):
            if item.is_dir():
                self.get_model(item.name)
