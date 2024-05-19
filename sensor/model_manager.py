import os

from common import ModelMetadata, PredictionModel
from common.model_utils import save_model_as_tflite, load_model_from_tflite


class ModelManager:
    base_dir = os.path.abspath(os.path.dirname(__file__))

    def __init__(self, model_dir: str) -> None:
        self._model_dir = os.path.join(ModelManager.base_dir, model_dir)
        self._models: dict[str, PredictionModel] = {}

    def save_model(self, model_bytes: bytes, model_metadata: ModelMetadata) -> PredictionModel:
        """
        Saves a model into the Sensor's storage

        :param model_bytes: the bytes of the model to save
        :param model_metadata: the metadata of the model
        """
        model_name = model_metadata.model_id
        save_model_as_tflite(model_bytes, self._model_dir, model_name)
        model = load_model_from_tflite(self._model_dir, model_name, model_metadata)
        self._models[model_name] = model
        return model

    def get_model(self, model_metadata: ModelMetadata) -> PredictionModel:
        """
        Retrieves a model.

        :param model_metadata: the metadata of the model to load
        :return: the loaded prediction model
        """
        model_name = model_metadata.model_id
        model = self._models.get(model_name)
        if model is None:
            model = load_model_from_tflite(self._model_dir, model_name, model_metadata)
            self._models[model_name] = model
        return model
