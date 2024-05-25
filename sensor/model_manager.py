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
