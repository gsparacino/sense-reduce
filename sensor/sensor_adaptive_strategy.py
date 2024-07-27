import datetime
import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from common import ThresholdMetric, Predictor
from common.resource_profiler import profiled
from sensor.base_station_gateway import BaseStationGateway
from sensor.model_manager import ModelManager
from sensor.violation import Violation


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
                 violations_limit: int = 0,
                 ):
        self.threshold_metric = threshold_metric
        self.model_manager: ModelManager = model_manager
        self.base_station: BaseStationGateway = base_station
        self.cooldown = cooldown
        self._latest_model_switch_timestamp = datetime.datetime.now()
        self._violations_count = 0
        self.violations_limit = violations_limit

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

        logging.info(
            f"Threshold violation ({self._violations_count} / {self.violations_limit}): "
            f"Measurement={measurement}, Prediction={prediction}"
        )
        predictor.log_violation(timestamp)

        self._violations_count += 1

        # Still in cooldown, do nothing
        if self._in_cooldown(timestamp):
            return predictor

        if self._violations_count < self.violations_limit:
            # Still within the violation threshold, try to refresh the current model's Prediction Horizon, and check if
            # that's enough to solve the violation. If it does, keep the current model (but don't reset the violation
            # counter)
            predictor.update_prediction_horizon(timestamp)
            prediction = predictor.get_prediction_at(timestamp).to_numpy()
            if not threshold_metric.is_threshold_violation(measurement, prediction):
                logging.debug(
                    f"Refreshed prediction horizon after violation - current model: {predictor.model_id}"
                )
                return predictor

        # Counter is over the violation threshold: search for a better model in the Sensor's portfolio
        new_predictor = self._find_better_predictor(threshold_metric, predictor, timestamp, measurement)
        if new_predictor is not None:
            # Better model found, switch and reset the counter. Also, send reduced measurements to the BS and query for
            # portfolio updates
            logging.debug(f"Switching to new model: {new_predictor.model_id}")
            self._latest_model_switch_timestamp = timestamp
            measurements = predictor.get_reduced_measurements_in_current_prediction_horizon(timestamp)
            models = base_station.synchronize(node_id, timestamp, new_predictor.model_id, measurements)
            model_manager.synchronize_models(models)
            self._violations_count = 0
            return new_predictor

        # Better model not found, keep current model for now while asking the BS for a new one
        logging.debug(f"No suitable model found, requesting new model")
        violation_measurements = predictor.get_measurements_in_current_prediction_horizon(timestamp)
        portfolio = model_manager.get_models_in_portfolio()
        models = base_station.send_violation(
            node_id, timestamp, violation_measurements, predictor.model_id, portfolio, True
        )
        model_manager.synchronize_models(models)
        return predictor

    @profiled(tag="New model selection")
    def _find_better_predictor(self,
                               threshold_metric: ThresholdMetric,
                               current_predictor: Predictor,
                               timestamp: datetime.datetime,
                               measurements: np.array,
                               ) -> Optional[Predictor]:
        """
        Iterates over the provided PredictionModels, comparing their performance on the latest measurements and returning
        a Predictor with the best model, or None if no other model offers better performances than the current one.

        :param threshold_metric: the threshold metric used to rank the models
        :param current_predictor: the Predictor currently used by the Sensor
        :param timestamp: the timestamp when the latest measurements were taken
        :param measurements: the latest measurements

        :return: a Predictor with the best possible PredictionModel according to the threshold_metric, or None if none
        of the other models has better performances than the current one.
        """

        num_previous_measurements = 3  # current_predictor.model_metadata.output_length
        prediction_period = current_predictor.prediction_period
        current_model = self.model_manager.get_model(current_predictor.model_metadata)

        # Retrieves the reduced list of measurements within the last prediction horizon
        previous_measurements = (
            current_predictor.data.get_previous_measurements(timestamp, num_previous_measurements, prediction_period)
        )
        previous_timestamps = previous_measurements.index

        predictor = Predictor(current_model, prediction_period, current_predictor.data)
        predictor.update_prediction_horizon(previous_timestamps.min())
        best_score = 0
        for idx in previous_timestamps:
            prediction = predictor.get_prediction_at(idx)
            best_score += threshold_metric.threshold_score(measurements, prediction)

        best_predictor = None
        models = self.model_manager.get_models()
        for model in list(models.values()):
            model_id = model.metadata.model_id
            if model_id == current_predictor.model_id:
                continue
            predictor = Predictor(model, prediction_period, current_predictor.data)
            predictor.update_prediction_horizon(previous_timestamps.min())
            prediction = predictor.get_prediction_at(timestamp).to_numpy()
            if threshold_metric.is_threshold_violation(measurements, prediction):
                logging.debug(f"Model candidate {model_id} would have violated the threshold, skipping")
                continue
            score = 0
            for idx in previous_timestamps:
                prediction = predictor.get_prediction_at(idx)
                score += threshold_metric.threshold_score(measurements, prediction)

            logging.debug(f"Model candidate {model_id} score: {score} | score to beat: {best_score}")
            if score < best_score:
                logging.debug(f"New best model: {predictor.model_id}")
                best_score = score
                best_predictor = predictor

        if best_predictor is not None:
            best_predictor.update_prediction_horizon(timestamp)

        return best_predictor

    @profiled(tag="Synchronization")
    def _synchronize_with_base_station(self, node_id: str, predictor: Predictor, timestamp: datetime.datetime) -> None:
        """
        Synchronizes with the base station state by sending the latest measurements, and fetching or deleting local
        models to reflect the current state of the models' portfolio on the Base Station.

        :param timestamp: The timestamp of the synchronization, as a datetime.datetime.
        """
        latest_measurements = predictor.get_measurements_in_current_prediction_horizon(timestamp)
        latest_measurements.dropna(inplace=True)
        models = self.base_station.synchronize(node_id, timestamp, predictor.model_id, latest_measurements)
        self.model_manager.synchronize_models(models)

    def _in_cooldown(self, timestamp):
        latest_event = self._latest_model_switch_timestamp
        return latest_event is not None and (timestamp - latest_event) < self.cooldown
