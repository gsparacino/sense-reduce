import datetime
import logging
import time

import pandas as pd

from abstract_sensor import AbstractSensor
from common import ThresholdMetric, Predictor
from sensor.Violation import Violation
from sensor.adaptive_strategy import AdaptiveStrategy
from sensor.base_station_gateway import BaseStationGateway
from sensor.model_manager import ModelManager

NodeID = str


class SensorManager:

    def __init__(self,
                 node_id: NodeID,
                 sensor: AbstractSensor,
                 predictor: Predictor,
                 base_station_gateway: BaseStationGateway,
                 model_manager: ModelManager,
                 threshold_metric: ThresholdMetric,
                 cooldown: float,
                 ) -> None:
        """
        Manages a sensor's measurements and prediction models, as well as handling coordination with the Base Station.

        :param node_id: The node id of the sensor node
        :param sensor: An implementation of AbstractSensor that provides measurements.
        :param predictor: An instance of Predictor that takes measurements as inputs and provides predictions.
        :param base_station_gateway: An instance of BaseStationGateway that handles coordination with the Base Station.
        :param model_manager: An instance of ModelManager that manages the local portfolio of models.
        :param threshold_metric: A metric implementation that defines a threshold violation.
        :param cooldown: The minimum time interval between consecutive violations to which the sensor should react.
        """
        logging.debug(f"Initializing SensorManager")
        assert cooldown > 0, "cooldown must be greater than 0."

        self.node_id = node_id
        self.sensor: AbstractSensor = sensor
        self.predictor: Predictor = predictor
        self.model_manager: ModelManager = model_manager
        self.base_station_gateway: BaseStationGateway = base_station_gateway
        self._latest_violation_timestamp = None
        self._cooldown = datetime.timedelta(seconds=cooldown)
        self._adaptive_strategy = (
            AdaptiveStrategy(threshold_metric, model_manager, base_station_gateway)
        )
        self._threshold_metric = threshold_metric

    def run(self, time_interval: float) -> None:
        """Starts monitoring the sensor, coordinating with the Base Station by sending data notifying violations.
        """
        assert time_interval > 0, "time_interval must be greater than 0."

        node_id = self.node_id

        while True:
            measurement, timestamp = self._get_measurement()
            logging.debug(f"Measurement @ {timestamp}: {measurement.values}")

            measurements_array = measurement.to_numpy()
            self.predictor.add_measurement(timestamp, measurements_array)

            if not self.predictor.in_prediction_horizon(timestamp):
                self.predictor.update_prediction_horizon(timestamp)
            prediction = self._get_prediction(timestamp)
            logging.debug(f"Prediction by {self.predictor.model_id} @ {timestamp}: {prediction.values}")

            prediction_array = prediction.to_numpy()
            self.predictor.add_prediction(timestamp, prediction_array)

            if self._adaptive_strategy.is_violation(measurements_array, prediction_array):
                if self._not_in_cooldown(timestamp):
                    self._latest_violation_timestamp = timestamp
                    violation = Violation(node_id, timestamp, self.predictor, measurements_array, prediction_array)
                    self.predictor = self._adaptive_strategy.handle_violation(violation)
                else:
                    logging.debug(
                        f"Threshold violation happened within the cooldown period. Ignoring violations until "
                        f"{self._latest_violation_timestamp + self._cooldown}"
                    )

            time.sleep(time_interval)

    def _not_in_cooldown(self, timestamp):
        latest_violation = self._latest_violation_timestamp
        return latest_violation is None or (timestamp - latest_violation) > self._cooldown

    def _get_prediction(self, timestamp: datetime.datetime):
        predictor = self.predictor
        prediction = predictor.get_prediction_at(timestamp)
        predictor.add_prediction(timestamp, prediction.to_numpy())
        return prediction

    def _get_measurement(self) -> (pd.Series, datetime):
        now = datetime.datetime.now()
        measurement = self.sensor.measurement
        while measurement is None:
            now = datetime.datetime.now()
            logging.warning(f'{now} - Failed reading measurement from sensor. Retrying...')
            measurement = self.sensor.measurement
        return measurement, now
