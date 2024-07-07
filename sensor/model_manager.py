import datetime
import logging
import os
from typing import Optional

import numpy as np

from common import ModelMetadata, PredictionModel, ThresholdMetric, Predictor
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
        self._model_rankings: dict[str, int] = {}
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
        expected_models = set(model_ids)

        models_to_remove = current_models.difference(expected_models)
        for model_id in models_to_remove:
            self._delete_model(model_id)
        models_to_add = expected_models.difference(current_models)
        for model_id in models_to_add:
            model_metadata = self._base_station.get_model_metadata(node_id=self.node_id, model_id=model_id)
            self.add_model(model_metadata)

    def _delete_model(self, model_name: str) -> None:
        """
        Removes a Model from the Sensor's portfolio, if present.

        :param model_name: The name of the model to delete
        """
        model_to_delete = self._models.pop(model_name)
        if model_to_delete:
            path = get_model_dir_path(self._model_dir, model_name)
            delete_tflite_model(path)

    def get_better_predictor(self, threshold_metric: ThresholdMetric,
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

        :return: a Predictor with the best possible PredictionModel according to the threshold_metric, or None if none
        of the models has better performances than the current one.
        """
        best_predictor = self._get_better_predictor(
            threshold_metric, current_predictor, timestamp, measurements, prediction
        )
        return best_predictor

    def _load_local_models(self) -> None:
        """
        Initializes the Sensor's models portfolio by loading all TFLite models in its storage.
        """
        for item in os.scandir(self._model_dir):
            if item.is_dir():
                model = load_model_tflite(item.path)
                self._models[model.metadata.model_id] = model

    def _get_better_predictor(self,
                              threshold_metric: ThresholdMetric,
                              current_predictor: Predictor,
                              timestamp: datetime.datetime,
                              measurements: np.array,
                              prediction: np.array) -> Optional[Predictor]:
        """
        Iterates over the provided PredictionModels, comparing their performance on the latest measurements and returning
        a Predictor with the best model.

        :param threshold_metric: the threshold metric used to rank the models
        :param current_predictor: the Predictor currently used by the Sensor
        :param timestamp: the timestamp when the latest measurements were taken
        :param measurements: the latest measurements
        :param prediction: the latest prediction

        :return: a Predictor with the best possible PredictionModel according to the threshold_metric, or None if none
        of the models has better performances than the current one.
        """
        best_score = threshold_metric.threshold_score(measurements, prediction)
        best_predictor = None
        for model in list(self._models.values()):
            model_id = model.metadata.model_id
            if model_id == current_predictor.model_id:
                continue
            predictor = Predictor(model, current_predictor.prediction_period, current_predictor.data)
            predictor.update_prediction_horizon(timestamp)
            prediction = predictor.get_prediction_at(timestamp).to_numpy()
            if threshold_metric.is_threshold_violation(measurements, prediction):
                logging.debug(f"Model candidate {model_id} would have violated the threshold, skipping")
                continue
            score = threshold_metric.threshold_score(measurements, prediction)
            logging.debug(f"Model candidate {model_id} score: {score} | score to beat: {best_score}")
            if score < best_score:
                logging.debug(f"New best model: {predictor.model_id}")
                best_score = score
                best_predictor = predictor
        return best_predictor
