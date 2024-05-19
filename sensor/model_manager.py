import os

from common import ModelMetadata, PredictionModel
from common.model_utils import save_model_as_tflite, load_model_from_tflite


class ModelManager:
    base_dir = os.path.abspath(os.path.dirname(__file__))

    def __init__(self, model_dir: str) -> None:
        self._model_dir = os.path.join(ModelManager.base_dir, model_dir)

    def save_model(self, model_bytes: bytes, model_name: str) -> None:
        """
        Saves a model into the Sensor's storage

        :param model_bytes: the bytes of the model to save
        :param model_name: the ID of the model
        """
        save_model_as_tflite(model_bytes, self._model_dir, model_name)

    def load_model(self, model_name: str, model_metadata: ModelMetadata) -> PredictionModel:
        """
        Loads a model from the Sensor's storage

        :param model_name: the ID of the model to load
        :param model_metadata: the metadata of the model to load
        :return: the loaded prediction model
        """
        return load_model_from_tflite(self._model_dir, model_name, model_metadata)
