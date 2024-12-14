import os
import shutil
from typing import Dict, List

from base.model import Model
from common.model_utils import load_model_from_savemodel, save_model, get_model_tflite_path, \
    get_model_dir_path


class ModelManager:

    def __init__(
            self,
            model_dir: str,
            base_model_id: str,
    ):
        """
        Handles I/O operations for Models.
        """
        self._model_dir = model_dir
        self._models: Dict[str, Model] = {}
        self._load_local_models()
        self.base_model: Model = self.get_model(base_model_id)

    def get_all_model_ids(self) -> List[str]:
        """
        :return: the list of IDs of all available models.
        """
        models: List[str] = list(model_id for model_id in self._models.keys())
        return models

    def get_all_models(self) -> List[Model]:
        """
        :return: all available models.
        """
        return list(self._models.values())

    def get_model(self, model_id: str) -> Model:
        """
        Returns the model with the provided id

        :param model_id: The ID of the model

        :return: The requested model

        :raises ValueError: if the provided str does not match any model
        """
        model = self._models.get(model_id)
        if model is None:
            model = self._load_model(model_id)
            if model is None:
                raise ValueError(f'Model {model_id} not found')
            self._models[model_id] = model
        return model

    def get_models(self, model_ids: set[str]) -> set[Model]:
        """
        Returns the models with the provided ids

        :param model_ids: The IDs of the models to retrieve

        :return: The requested models

        :raises ValueError: if any of the provided ids does not match any model
        """
        result = set()
        for model_id in model_ids:
            model = self.get_model(model_id)
            result.add(model)
        return result

    def _load_model(self, model_id: str) -> Model:
        model_path = os.path.join(self._model_dir, model_id)
        model = load_model_from_savemodel(model_path)
        return model

    def get_model_tflite_file_path(self, model_id: str) -> os.path:
        path = get_model_dir_path(self._model_dir, model_id)
        return get_model_tflite_path(path, model_id)

    def save_model(self, model: Model) -> Model:
        """
        Saves a model as a file in the base station's model directory.

        :param model: the model to save
        """
        model_id = model.metadata.uuid
        path = get_model_dir_path(self._model_dir, model_id)
        save_model(model, path)
        return self.get_model(model_id)

    def delete_model(self, model_id: str, hard_delete: bool = False) -> None:
        """
        Removes a model from the base station's collection.

        Args:
            model_id: the ID of the model to delete
            hard_delete: if the model should be deleted permanently from the base station's storage
        """
        if model_id in self._models:
            self._models.pop(model_id)
            if hard_delete:
                path = get_model_dir_path(self._model_dir, model_id)
                shutil.rmtree(path)

    def _load_local_models(self) -> None:
        """
        Initializes the Sensor's models portfolio by loading all TFLite models in its storage.
        """
        for item in os.scandir(self._model_dir):
            if item.is_dir():
                self.get_model(item.name)
