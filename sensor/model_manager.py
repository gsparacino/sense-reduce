import logging
import os
from typing import Optional

from common.lite_model import LiteModel
from common.model_metadata import ModelMetadata
from common.model_utils import save_model_as_tflite, load_model_tflite, delete_tflite_model, \
    get_model_dir_path
from common.prediction_model import PredictionModel
from sensor.base_station_gateway import BaseStationGateway


class ModelManager:

    def __init__(self, model_dir: str, base_station: BaseStationGateway):
        """
        Manages a portfolio of PredictionModels.

        :param model_dir: The local directory in which the models are stored.
        :param base_station: A BaseStationGateway instance, used to interact with the Node's Base Station.
        """
        self._base_station = base_station
        self._model_dir = model_dir
        self._models: dict[str, LiteModel] = {}
        self._load_local_models()

    def _save_model(self, model_bytes: bytes, metadata: ModelMetadata) -> LiteModel:
        """
        Saves a model into the Sensor's storage

        :param model_bytes: the bytes of the model to save
        :param metadata: the metadata of the model
        """
        model_name = metadata.uuid
        model_dir = get_model_dir_path(self._model_dir, model_name)
        save_model_as_tflite(model_bytes, metadata, model_dir)
        model = load_model_tflite(model_dir)
        self._models[model_name] = model
        return model

    def get_model(self, model_id: str) -> Optional[PredictionModel]:
        """
        Retrieves a model from the Sensor's portfolio, if present; otherwise returns None.

        :param model_id: The ID of the Model to load
        :return: The PredictionModel, or None if the model is not found
        """
        model = self._models.get(model_id)
        return model

    def get_models(self) -> dict[str, PredictionModel]:
        """
        :return: a dictionary of all available models in the Sensor's portfolio. The keys of the dictionary represent
        the models' unique IDs.
        """
        return self._models

    def get_model_ids(self) -> list[str]:
        """
        :return: the list of IDs of all models in the portfolio.
        """
        return list(model.metadata.uuid for model in self._models.values())

    def add_model(self, metadata: ModelMetadata) -> LiteModel:
        """
        Adds a new model to the Sensor's portfolio, fetching it from the Base Station if necessary.

        :param metadata: The metadata of the Model to add

        :return: The new PredictionModel
        """
        model_name = metadata.uuid
        model = self._models.get(model_name)
        if model is None:
            model_bytes: bytes = self._base_station.fetch_model(model_name)
            model = self._save_model(model_bytes, metadata)
        return model

    def synchronize_models(self, model_ids: set[str]) -> set[str]:
        """
        Updates the list of Sensor's models by removing local models that are not present in the provided list, and
        adding those models that are present in the provided list but not in the Sensor's portfolio.

        Args:
            model_ids: the updated list of models received from the BS

        Returns:
            The list of new models that were added to the Sensor's portfolio.
        """
        current_models = set(model_id for model_id in self._models.keys())
        logging.debug(f"Portfolio update - current models: {self._model_list(current_models)}")
        logging.debug(f"Portfolio update - target models: {self._model_list(model_ids)}")

        models_to_remove = current_models.difference(model_ids)
        for model_id in models_to_remove:
            self._delete_model(model_id)
        models_to_add = model_ids.difference(current_models)
        for model_id in models_to_add:
            model_metadata = self._base_station.get_model_metadata(model_id=model_id)
            self.add_model(model_metadata)
        return models_to_add

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
                self._models[model.metadata.uuid] = model
