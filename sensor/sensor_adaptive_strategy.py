import datetime
import logging
from abc import ABC, abstractmethod

import numpy as np

from common import ThresholdMetric, Predictor
from sensor.Violation import Violation
from sensor.base_station_gateway import BaseStationGateway
from sensor.model_manager import ModelManager


class SensorNodeAdaptiveStrategy(ABC):

    @abstractmethod
    def is_violation(self, measurement: np.array, prediction: np.array) -> bool:
        """
        Tests whether a measurement is incompatible with the corresponding prediction, signaling that the current
        Predictor may be inadequate.

        :param measurement: the measurement to be tested
        :param prediction: the prediction to compare the measurement against
        """
        pass

    @abstractmethod
    def handle_violation(self, violation: Violation) -> Predictor:
        """
        Handles a violation, providing an updated Predictor.

        :param violation: the violation data
        :return: an updated Predictor
        """
        pass


class DefaultSensorNodeAdaptiveStrategy(SensorNodeAdaptiveStrategy):
    def __init__(self,
                 threshold_metric: ThresholdMetric,
                 model_manager: ModelManager,
                 base_station: BaseStationGateway,
                 cooldown: datetime.timedelta,
                 ):
        self.threshold_metric = threshold_metric
        self.model_manager: ModelManager = model_manager
        self.base_station: BaseStationGateway = base_station
        self.cooldown = cooldown
        self._latest_model_switch_timestamp = None

    def is_violation(self, measurement: np.array, prediction: np.array) -> bool:
        return self.threshold_metric.is_threshold_violation(measurement, prediction)

    def handle_violation(self, violation: Violation) -> Predictor:
        """
        Handles a violation, providing an updated Predictor.

        :param violation: the violation data
        :return: an update Predictor
        """
        threshold_metric = self.threshold_metric
        base_station = self.base_station
        model_manager = self.model_manager
        node_id = violation.node_id
        timestamp = violation.timestamp
        measurement = violation.measurement
        prediction = violation.prediction
        predictor = violation.predictor

        if self._not_in_cooldown(timestamp):
            logging.info(
                f"Threshold violation: Measurement={measurement}, Prediction={prediction}"
            )
            new_predictor = model_manager.get_better_predictor(
                threshold_metric, predictor, timestamp, measurement, prediction
            )
            request_new_model = False
            if new_predictor is not None:
                logging.debug(f"Switching to new model: {new_predictor.model_id}")
                self._latest_model_switch_timestamp = timestamp
            else:
                new_predictor = predictor
                logging.debug(f"No suitable model found, requesting new model")
                request_new_model = True
                new_predictor.add_violation(timestamp)
            violation_measurement = predictor.get_measurement(timestamp)
            portfolio = model_manager.get_models_in_portfolio()
            models = base_station.send_violation(
                node_id, timestamp, violation_measurement, predictor.model_id, portfolio, request_new_model
            )
            model_manager.synchronize_models(models)
            return new_predictor
        else:
            logging.debug(
                f"Threshold violation happened within the cooldown period. Ignoring violations until "
                f"{self._latest_model_switch_timestamp + self.cooldown}"
            )
            return predictor

    def _synchronize_with_base_station(self, node_id: str, predictor: Predictor, timestamp: datetime.datetime) -> None:
        """
        Synchronizes with the base station state by sending the latest measurements, and fetching or deleting local
        models to reflect the current state of the models' portfolio on the Base Station.

        :param timestamp: The timestamp of the synchronization, as a datetime.datetime.
        """
        model_manager = self.model_manager

        latest_measurements = predictor.get_measurements_in_current_prediction_horizon(timestamp)
        models = self.base_station.synchronize(node_id, timestamp, predictor.model_id, latest_measurements)
        model_manager.synchronize_models(models)

    def _not_in_cooldown(self, timestamp):
        latest_event = self._latest_model_switch_timestamp
        return latest_event is None or (timestamp - latest_event) > self.cooldown
