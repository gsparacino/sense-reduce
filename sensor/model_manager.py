import logging
import os
from typing import Optional

from common import ModelMetadata, PredictionModel
from common.model_utils import save_model_as_tflite, load_model_tflite, delete_tflite_model, \
    get_model_dir_path
from sensor.base_station_gateway import BaseStationGateway


class ModelManager:
    base_dir = os.path.abspath(os.path.dirname(__file__))

    def __init__(self, node_id: str, model_dir: str, base_station_gateway: BaseStationGateway):
        """
        Manages the Sensor's portfolio of PredictionModels.

        :param node_id: The Sensor's unique ID within its cluster.
        :param model_dir: The local directory in which the models are stored.
        :param base_station_gateway: A BaseStationGateway instance, used to interact with the Node's Base Station.
        """
        self.node_id = node_id
        self._base_station = base_station_gateway
        self._model_dir = os.path.join(ModelManager.base_dir, model_dir)
        self._models: dict[str, PredictionModel] = {}
        self._load_local_models()

    def _save_model(self, model_bytes: bytes, metadata: ModelMetadata) -> PredictionModel:
        """
        Saves a model into the Sensor's storage

        :param model_bytes: the bytes of the model to save
        :param metadata: the metadata of the model
        """
        model_name = metadata.model_id
        model_dir = get_model_dir_path(self._model_dir, model_name)
        save_model_as_tflite(model_bytes, metadata, model_dir)
        model = load_model_tflite(model_dir)
        self._models[model_name] = model
        return model

    def get_model(self, metadata: ModelMetadata) -> Optional[PredictionModel]:
        """
        Retrieves a model from the Sensor's portfolio, if present; otherwise returns None.

        :param metadata: The ModelMetadata of the Model to load
        :return: The PredictionModel, or None if the model is not found
        """
        model_name = metadata.model_id
        model = self._models.get(model_name)
        return model

    def get_models(self) -> dict[str, PredictionModel]:
        """
        :return: a dictionary of all available models in the Sensor's portfolio. The keys of the dictionary represent
        the models' unique IDs.
        """
        return self._models

    def get_models_in_portfolio(self) -> list[str]:
        """
        :return: the list of IDs of all models in the portfolio.
        """
        return list(model.metadata.model_id for model in self._models.values())

    def add_model(self, metadata: ModelMetadata) -> PredictionModel:
        """
        Adds a new model to the Sensor's portfolio, fetching it from the Base Station.

        :param metadata: The ModelMetadata of the Model to add

        :return: The new PredictionModel
        """
        model_name = metadata.model_id
        model = self._models.get(model_name)
        if model is None:
            model_bytes: bytes = self._base_station.get_model(self.node_id, model_name)
            model = self._save_model(model_bytes, metadata)
        return model

    def synchronize_models(self, model_ids: list[str]) -> None:
        """
        Updates the list of Sensor's models by removing local models that are not present in the provided list, and
        adding those models that are present in the provided list but not in the Sensor's portfolio.

        :param model_ids: The list of model IDs that are available for the node
        """
        current_models = set(model_id for model_id in self._models.keys())
        logging.debug(f"Portfolio update - current models: {self._model_list(current_models)}")
        expected_models = set(model_ids)
        logging.debug(f"Portfolio update - BS models: {self._model_list(expected_models)}")

        models_to_remove = current_models.difference(expected_models)
        logging.debug(f"Portfolio update - models to remove: {self._model_list(models_to_remove)}")
        for model_id in models_to_remove:
            self._delete_model(model_id)
        models_to_add = expected_models.difference(current_models)
        logging.debug(f"Portfolio update - models to add: {self._model_list(models_to_add)}")
        for model_id in models_to_add:
            model_metadata = self._base_station.get_model_metadata(node_id=self.node_id, model_id=model_id)
            self.add_model(model_metadata)

    @staticmethod
    def _model_list(models: set[str]) -> str:
        result = "["
        f_comma = False
        for model in models:
            if f_comma:
                result += ", "
            else:
                f_comma = True
            result += model
        return result + "]"

    def _delete_model(self, model_name: str) -> None:
        """
        Removes a Model from the Sensor's portfolio, if present.

        :param model_name: The name of the model to delete
        """
        model_to_delete = self._models.pop(model_name)
        if model_to_delete:
            path = get_model_dir_path(self._model_dir, model_name)
            delete_tflite_model(path)

    def _load_local_models(self) -> None:
        """
        Initializes the Sensor's models portfolio by loading all TFLite models in its storage.
        """
        for item in os.scandir(self._model_dir):
            if item.is_dir():
                model = load_model_tflite(item.path)
                self._models[model.metadata.model_id] = model
