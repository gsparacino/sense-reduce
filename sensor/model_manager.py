import datetime
import logging
import os
from typing import Optional

import numpy as np

from common import ModelMetadata, PredictionModel, ThresholdMetric, Predictor
from common.model_utils import save_model_as_tflite, load_model_from_tflite


class ModelManager:
    base_dir = os.path.abspath(os.path.dirname(__file__))

    def __init__(self, model_dir: str) -> None:
        self._model_dir = os.path.join(ModelManager.base_dir, model_dir)
        self._models: dict[str, PredictionModel] = {}
        self._initialize_models()

    def save_model(self, model_bytes: bytes, metadata: ModelMetadata) -> PredictionModel:
        """
        Saves a model into the Sensor's storage

        :param model_bytes: the bytes of the model to save
        :param metadata: the metadata of the model
        """
        model_name = metadata.model_id
        model_dir = os.path.join(self._model_dir, model_name)
        save_model_as_tflite(model_bytes, metadata, model_dir)
        model = load_model_from_tflite(model_dir)
        self._models[model_name] = model
        return model

    def get_model_from_portfolio(self, model_name: str) -> Optional[PredictionModel]:
        """
        Retrieves a model from the Sensor's portfolio, if present; otherwise returns None.

        :param model_name: The name of the model to load
        :return: The PredictionModel, or None if the model is not found
        """
        return self._models.get(model_name)

    def get_new_predictor(self, threshold_metric: ThresholdMetric,
                          current_predictor: Predictor,
                          timestamp: datetime.datetime,
                          measurements: np.array,
                          prediction: np.array) -> Optional[Predictor]:
        """
        Iterates over the available PredictionModels, compares their performance on the latest measurements and returns
        a Predictor with the best model.

        :param threshold_metric: the threshold metric used to rank the models
        :param current_predictor: the Predictor currently used by the Sensor
        :param timestamp: the timestamp when the latest measurements were taken
        :param measurements: the latest measurements
        :param prediction: the latest prediction
        :return: a Predictor with the best model found
        """
        best_score = threshold_metric.threshold_score(measurements, prediction)
        best_predictor = None
        for model_name, model in self._models.items():
            if model_name == current_predictor.model_id:
                continue
            predictor = Predictor(model, current_predictor.data, current_predictor.prediction_period)
            predictor.update_prediction_horizon(timestamp)
            prediction = predictor.get_prediction_at(timestamp).to_numpy()
            score = threshold_metric.threshold_score(measurements, prediction)
            logging.debug(f"{predictor.model_id} score: {best_score}")
            if score < best_score:
                best_score = score
                best_predictor = predictor
                logging.debug(f"New best model: {predictor.model_id}")

        return best_predictor

    def _initialize_models(self) -> None:
        """
        Initializes the Sensor's models portfolio by loading all TFLite models in its storage.
        """
        for item in os.scandir(self._model_dir):
            if item.is_dir():
                path = item.path
                model = load_model_from_tflite(path)
                self._models[model.metadata.model_id] = model
