import datetime
import logging
import time

import pandas as pd

from abstract_sensor import AbstractSensor
from common import ThresholdMetric, Predictor
from sensor.base_station_gateway import BaseStationGateway
from sensor.model_manager import ModelManager

NodeID = str


class SensorManager:

    def __init__(self,
                 node_id: NodeID,
                 sensor: AbstractSensor,
                 predictor: Predictor,
                 base_station_gateway: BaseStationGateway,
                 model_manager: ModelManager
                 ) -> None:
        """
        Manages a sensor's measurements and prediction models, as well as handling coordination with the Base Station.

        :param node_id: The node id of the sensor node
        :param sensor: An implementation of AbstractSensor that provides measurements.
        :param predictor: An instance of Predictor that takes measurements as inputs and provides predictions.
        :param base_station_gateway: An instance of BaseStationGateway that handles coordination with the Base Station.
        :param mode: The sensor's initial operating mode.
        """
        logging.debug(f"Initializing SensorManager")
        self.node_id = node_id
        self.sensor: AbstractSensor = sensor
        self.predictor: Predictor = predictor
        self.model_manager: ModelManager = model_manager
        self.base_station_gateway: BaseStationGateway = base_station_gateway
        self._latest_violation_timestamp = None

    def run(self, threshold_metric: ThresholdMetric, time_interval: float, cooldown: float) -> None:
        """Starts monitoring the sensor, coordinating with the Base Station by sending data notifying violations.

        :param threshold_metric: A metric implementation that defines a threshold violation.
        :param time_interval: A float that represents the time period between consecutive measurements, in seconds.
        :param cooldown: The minimum time interval between consecutive violations to which the sensor should react.
        """
        assert time_interval > 0, "time_interval must be greater than 0."
        assert cooldown > 0, "cooldown must be greater than 0."
        cooldown = datetime.timedelta(seconds=cooldown)

        base_station = self.base_station_gateway
        node_id = self.node_id

        while True:
            measurement, timestamp = self._get_measurement()
            logging.debug(f"Measurement @ {timestamp}: {measurement.values}")

            measurements_array = measurement.to_numpy()
            self.predictor.add_measurement(timestamp, measurements_array)

            if not self.predictor.in_prediction_horizon(timestamp):
                self._synchronize_with_base_station(timestamp)
                self.predictor.update_prediction_horizon(timestamp)
            prediction = self._get_prediction(timestamp)
            logging.debug(f"Prediction by {self.predictor.model_id} @ {timestamp}: {prediction.values}")

            prediction_array = prediction.to_numpy()
            self.predictor.add_prediction(timestamp, prediction_array)

            if threshold_metric.is_threshold_violation(measurements_array, prediction_array):
                latest_violation_timestamp = self.predictor.get_latest_violation_datetime()
                if latest_violation_timestamp is None or (timestamp - latest_violation_timestamp) > cooldown:
                    logging.info(
                        f"Threshold violation: Measurement={measurement.values}, Prediction={prediction.values}"
                    )
                    new_predictor = self.model_manager.get_better_predictor(
                        threshold_metric, self.predictor, timestamp, measurements_array, prediction
                    )
                    request_new_model = False
                    if new_predictor is not None:
                        self.predictor = new_predictor
                        logging.debug(f"Switching to new model: {new_predictor.model_id}")
                    else:
                        logging.debug(f"No suitable model found, requesting new model")
                        request_new_model = True
                        self.predictor.add_violation(timestamp)
                    violation_measurement = self.predictor.get_measurement(timestamp)
                    portfolio = self.model_manager.get_models_in_portfolio()
                    models = base_station.send_violation(
                        node_id, timestamp, violation_measurement, self.predictor.model_id, portfolio, request_new_model
                    )
                    self.model_manager.synchronize_models(models)

                else:
                    logging.debug(
                        f"Threshold violation within the cooldown period, ignoring violations until "
                        f"{latest_violation_timestamp + cooldown}"
                    )

            time.sleep(time_interval)

    def _get_prediction(self, timestamp: datetime.datetime):
        predictor = self.predictor
        prediction = predictor.get_prediction_at(timestamp)
        predictor.add_prediction(timestamp, prediction.to_numpy())
        return prediction

    def _synchronize_with_base_station(self, timestamp: datetime.datetime):
        """
        Synchronizes with the base station state by sending the latest measurements, and fetching or deleting local
        models to reflect the current state of the models' portfolio on the Base Station.

        :param timestamp: The timestamp of the synchronization, as a datetime.datetime.
        """
        predictor = self.predictor
        latest_measurements = predictor.get_measurements_in_current_prediction_horizon(timestamp)
        models = self.base_station_gateway.synchronize(self.node_id, timestamp, latest_measurements)
        self.model_manager.synchronize_models(models)

    def _get_measurement(self) -> (pd.Series, datetime):
        now = datetime.datetime.now()
        measurement = self.sensor.measurement
        while measurement is None:
            now = datetime.datetime.now()
            logging.warning(f'{now} - Failed reading measurement from sensor. Retrying...')
            measurement = self.sensor.measurement
        return measurement, now
