import datetime
import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd

from common import ThresholdMetric, Predictor, PredictionHorizon, ModelMetadata, DataStorage
from common.resource_profiler import profiled
from common.violation_monitor import ViolationsMonitor
from sensor.base_station_gateway import BaseStationGateway, NodeInitialization
from sensor.model_manager import ModelManager
from sensor.violation import Violation


class SensorNodeAdaptiveStrategy(ABC):

    @abstractmethod
    def execute(self, measurement: np.array, timestamp: datetime.datetime) -> None:
        """
        Executes the Strategy, handling the new measurement and adapting the Sensor's configurations if necessary.

        Args:
            measurement: the latest measurement read by the Sensor
            timestamp: the timestamp of the measurement
        """
        pass

    @profiled(tag="Synchronization")
    def _synchronize_with_base_station(self,
                                       base_station: BaseStationGateway,
                                       model_manager: ModelManager,
                                       node_id: str,
                                       predictor: Predictor,
                                       timestamp: datetime.datetime
                                       ) -> None:
        """
        Synchronizes with the base station by sending the latest measurements, and fetching or deleting local
        models to reflect the current state of the models' portfolio on the Base Station.

        :param timestamp: The timestamp of the synchronization, as a datetime.datetime.
        """
        latest_measurements = predictor.get_reduced_measurements_in_current_prediction_horizon(timestamp)
        models = base_station.synchronize(node_id, timestamp, predictor.model_id, latest_measurements)
        model_manager.synchronize_models(models)


class DefaultSensorNodeAdaptiveStrategy(SensorNodeAdaptiveStrategy):
    def __init__(self,
                 node_id: str,
                 model_metadata: ModelMetadata,
                 threshold_metric: ThresholdMetric,
                 model_manager: ModelManager,
                 base_station: BaseStationGateway,
                 violations_monitor: ViolationsMonitor,
                 data_storage: DataStorage,
                 prediction_interval: float
                 ):
        """
        The default AdaptiveStrategy for a Sensor Node.

        Args:
            node_id: the unique ID of the node
            model_metadata: the metadata of the initial model to be set for the strategy
            threshold_metric: the metric that defines a threshold violation
            model_manager: the helper that handles I/O operations for prediction models
            base_station: the helper that handles I/O operations with the base station
            violations_monitor: the helper that defines the suitability of models by monitoring the number of violations
            data_storage: the helper that handles I/O operations with the data storage
            prediction_interval: the time interval in seconds between consecutive data points in a Prediction Horizon
        """
        self.node_id = node_id
        self.threshold_metric = threshold_metric
        self.model_manager: ModelManager = model_manager
        self.base_station: BaseStationGateway = base_station
        self._violations_monitor = violations_monitor
        self.data_storage: DataStorage = data_storage
        self.prediction_interval = prediction_interval
        self.predictor = self._init_predictor(model_metadata, data_storage)

    def execute(self, measurement: np.array, timestamp: datetime.datetime) -> None:
        self.data_storage.add_measurement(timestamp, measurement)

        if self._can_make_predictions():

            if not self.predictor.in_prediction_horizon(timestamp):
                self._refresh_expired_horizon(timestamp)

            prediction = self.predictor.get_prediction_at(timestamp)
            logging.debug(f"Prediction by {self.predictor.model_id} @ {timestamp}: {prediction.values}")
            prediction_array = prediction.to_numpy()
            measurement_array = measurement.to_numpy()
            self.predictor.log_prediction(timestamp, prediction_array)

            if self.is_violation(measurement_array, prediction_array):
                violation = Violation(self.node_id, timestamp, self.predictor, measurement_array, prediction_array)
                self.predictor.add_violation(timestamp)
                self.handle_violation(violation)

        else:
            self.base_station.send_measurement(self.node_id, timestamp, measurement)

    def _can_make_predictions(self):
        min_measurements = self.predictor.model_metadata.input_length
        return len(self.data_storage.get_measurements()) >= min_measurements

    def _refresh_expired_horizon(self, timestamp):
        measurements = self.predictor.get_reduced_measurements_in_current_prediction_horizon(timestamp)
        self.base_station.synchronize(self.node_id, timestamp, self.predictor.model_id, measurements)
        self.predictor.add_prediction_horizon_update(timestamp)
        self.predictor.update_prediction_horizon(timestamp)
        logging.debug(f"Updated Prediction Horizon (scheduled update): \n {self.predictor.get_prediction_horizon()}")

    def _init_predictor(self,
                        model_metadata: ModelMetadata,
                        data_storage: DataStorage) -> Predictor:
        current_model = self.model_manager.get_model(model_metadata)
        period = datetime.timedelta(seconds=self.prediction_interval)
        predictor = Predictor(current_model, period, data_storage)
        predictor.update_prediction_horizon(datetime.datetime.now())
        return predictor

    def is_violation(self, measurement: np.array, prediction: np.array) -> bool:
        """
        Checks whether the discrepancies between the provided prediction and measurement constitute a violation,
        according to the threshold metric configured for this sensor.

        Args:
            measurement: the measurement to check
            prediction: the prediction associated with the measurement

        Returns:
            True if the difference between measurement and prediction constitutes a threshold violation, False otherwise

        """
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
        violations_monitor = self._violations_monitor

        # Send violation to BS, update portfolio
        latest_measurements = predictor.get_reduced_measurements_in_current_prediction_horizon(timestamp)
        portfolio = model_manager.get_models_in_portfolio()
        models = base_station.send_violation(
            node_id, timestamp, latest_measurements, predictor.model_id, portfolio, True
        )
        model_manager.synchronize_models(models)

        first_violation_timestamp = violations_monitor.get_first_violation_timestamp()
        violations_threshold_exceeded = violations_monitor.add_violation(timestamp)
        logging.info(
            f"Threshold violation ("
            f"{violations_monitor.get_violations_count()}/{violations_monitor.get_violations_limit()}"
            f" since {first_violation_timestamp if first_violation_timestamp is not None else '-'}): "
            f"Measurement={measurement}, Prediction={prediction}"
        )

        if not violations_threshold_exceeded:
            # Still within the violation monitor's threshold, try to refresh the current model's Prediction Horizon,
            # and check if that puts the predictions back on track. If it does, keep the current model, but don't reset
            # the violation counter: that should prevent too many consecutive updates of the prediction horizon.
            logging.debug(f'Refreshing Prediction Horizon at {timestamp}')
            predictor.update_prediction_horizon(timestamp)
            # diff = measurement[predictor.model_metadata.input_to_output_indices] - prediction
            # predictor.adjust_prediction_horizon(diff)
            logging.debug(
                f"Adjusted Prediction Horizon: \n "
                f"{predictor.get_prediction_horizon()}"
            )
            prediction = predictor.get_prediction_at(timestamp).to_numpy()
            if not threshold_metric.is_threshold_violation(measurement, prediction):
                return predictor
            else:
                logging.debug(f"Threshold violation despite Prediction Horizon update, current model is inadequate")
        else:
            logging.debug(f"Violations threshold exceeded, current model is inadequate")

        # Search for a better model in the Sensor's portfolio
        logging.debug(f"Attempting model switch")
        # Retrieves the reduced list of measurements within the last prediction horizon
        new_predictor = self._find_better_predictor(threshold_metric, predictor, first_violation_timestamp)
        if new_predictor is not None:
            # Better model found, switch and reset the counter. Also, send reduced measurements to the BS and query for
            # portfolio updates
            logging.debug(f"Switching to new model: {new_predictor.model_id}")
            new_predictor.update_prediction_horizon(datetime.datetime.now())
            logging.debug(f"Updated Prediction Horizon after model switch: \n {new_predictor.get_prediction_horizon()}")
            violations_monitor.reset()
            return new_predictor

        # No suitable model found, forward anomalous measurements to the BS
        logging.debug(f"No suitable model found")
        return predictor

    @profiled(tag="New model selection")
    def _find_better_predictor(self,
                               threshold_metric: ThresholdMetric,
                               current_predictor: Predictor,
                               evaluate_since: datetime.datetime,
                               ) -> Optional[Predictor]:
        """
        Iterates over the provided PredictionModels, comparing their performance on latest measurements and returning
        a Predictor with the best model, or None if no other model offers better performances than the current one.
        The measurements used for the comparison are those included in the [evaluation_since, now()] interval.

        :param threshold_metric: the threshold metric used to rank the models
        :param current_predictor: the Predictor currently used by the Sensor
        :param evaluate_since: the timestamp measurements used to compare the models' predictions accuracy

        :return: a Predictor with the best possible PredictionModel according to the threshold_metric, or None if none
        of the other models has better performances than the current one.
        """

        prediction_period = current_predictor.prediction_period
        current_model = self.model_manager.get_model(current_predictor.model_metadata)
        measurements: pd.DataFrame = current_predictor.get_measurements_since(evaluate_since)
        logging.debug(f"Evaluating models using measurements between {measurements.index.min()} "
                      f"and {measurements.index.max()}")

        predictor = Predictor(current_model, prediction_period, current_predictor.data)
        best_score = self._get_predictor_score(predictor, measurements, threshold_metric)

        best_predictor = None
        models = self.model_manager.get_models()
        for model in list(models.values()):
            model_id = model.metadata.model_id
            if model_id == current_predictor.model_id:
                # Skip current model, which is the baseline. i.e., its score is the initial score to beat (best_score)
                continue
            predictor = Predictor(model, prediction_period, current_predictor.data)
            score = self._get_predictor_score(predictor, measurements, threshold_metric)

            logging.debug(f"Model candidate: {model_id} | score: {score} | score to beat: {best_score}")
            if score > best_score:
                logging.debug(f"New best model: {predictor.model_id}")
                best_score = score
                best_predictor = predictor
            if best_score == 1:
                logging.debug(f"Model {best_predictor.model_id} reached maximum score, stopping models evaluation")
                break

        return best_predictor

    @staticmethod
    def _get_predictor_score(predictor: Predictor, measurements: pd.DataFrame, threshold_metric: ThresholdMetric):
        """
        Calculates the Predictor's performance score by measuring its accuracy in predicting the correct values for the
        provided measurements.

        Args:
            predictor: the Predictor to evaluate
            measurements: the measurements to test the Predictor on
            threshold_metric: the ThresholdMetric that defines the discrepancy between a measurement and a prediction

        Returns:
            The score of the predictor, which depends on the actual implementation of ThresholdMetric.threshold_score
        """
        timestamps = measurements.index
        predictor.update_prediction_horizon(timestamps.min())
        diff = 0
        for idx in timestamps:
            if not predictor.in_prediction_horizon(idx):
                predictor.update_prediction_horizon(idx)
            prediction = predictor.get_prediction_at(idx)
            measurement = measurements.loc[idx].to_numpy()
            diff += threshold_metric.threshold_score(measurement, prediction)
        return 1 / (1 + diff)
