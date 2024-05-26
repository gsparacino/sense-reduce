import datetime
import logging
import time
from enum import Enum, auto

import pandas as pd

from abstract_sensor import AbstractSensor
from common import ThresholdMetric, Predictor
from sensor.base_station_gateway import BaseStationGateway
from sensor.model_manager import ModelManager

NodeID = str


class SensorManager:
    class OperatingMode(Enum):
        """
        The sensor's operating mode is used to determine the amount of data that the sensor can send to the base station.
        """

        DATA_REDUCTION = auto(),
        """
        Aims to reduce the 'chattiness' of the Sensor: as long as the difference between a measurement and its 
        corresponding prediction is below the configured threshold, the sensor will not send data to the Base Station.
        """

        MODEL_TRAINING = auto(),
        """
        The Sensor will send all measurements the Base Station, that will eventually provide a new model.
        """

    def __init__(self,
                 node_id: NodeID,
                 sensor: AbstractSensor,
                 predictor: Predictor,
                 base_station_gateway: BaseStationGateway,
                 model_manager: ModelManager,
                 mode: OperatingMode = OperatingMode.DATA_REDUCTION
                 ) -> None:
        """
        Manages a sensor's measurements and prediction models, as well as handling coordination with the Base Station.

        :param node_id: The node id of the sensor node
        :param sensor: An implementation of AbstractSensor that provides measurements.
        :param predictor: An instance of Predictor that takes measurements as inputs and provides predictions.
        :param base_station_gateway: An instance of BaseStationGateway that handles coordination with the Base Station.
        :param mode: The sensor's initial operating mode.
        """
        logging.debug(f"Initializing SensorManager for node: {node_id}")
        self.node_id = node_id
        self.sensor: AbstractSensor = sensor
        self.predictor: Predictor = predictor
        self.model_manager: ModelManager = model_manager
        self.base_station_gateway: BaseStationGateway = base_station_gateway
        self._operating_mode = mode
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

            if self._operating_mode == SensorManager.OperatingMode.DATA_REDUCTION:
                prediction = self._get_prediction(measurements_array, timestamp)
                logging.debug(f"Prediction by {self.predictor.model_id} @ {timestamp}: {prediction.values}")

                prediction_array = prediction.to_numpy()
                self.predictor.add_prediction(timestamp, prediction_array)

                if threshold_metric.is_threshold_violation(measurements_array, prediction_array):
                    if (self._latest_violation_timestamp is None or
                            timestamp - self._latest_violation_timestamp > cooldown):
                        self._latest_violation_timestamp = timestamp

                        logging.info(
                            f"Threshold violation: Measurement={measurement.values}, Prediction={prediction.values}"
                        )
                        new_predictor = self.model_manager.get_new_predictor(
                            threshold_metric, self.predictor, timestamp, measurements_array, prediction
                        )
                        if new_predictor is None:
                            self._operating_mode = SensorManager.OperatingMode.MODEL_TRAINING
                            logging.debug(f"Switching to operating mode: {self._operating_mode.name}")
                        else:
                            self.predictor = new_predictor
                            logging.debug(f"Switching to new model: {new_predictor.model_id}")
                    else:
                        logging.debug(
                            f"Threshold violation within the cooldown period, ignoring violations until "
                            f"{self._latest_violation_timestamp + cooldown}"
                        )

            if self._operating_mode == SensorManager.OperatingMode.MODEL_TRAINING:
                new_data = self.predictor.get_measurements_in_current_prediction_horizon(timestamp)
                new_model_metadata = base_station.request_new_model(node_id, timestamp, new_data)
                if new_model_metadata is not None:
                    logging.debug(f"Base Station provided a new model {new_model_metadata.model_id}")
                    model = self.model_manager.add_model(new_model_metadata)
                    self.predictor = Predictor(model, self.predictor.data, self.predictor.get_prediction_timedelta())
                    self._operating_mode = SensorManager.OperatingMode.DATA_REDUCTION
                    logging.debug(f"Switching to operating mode {self._operating_mode.name}")
                self.predictor.update_prediction_horizon(timestamp)

            time.sleep(time_interval)

    def _get_prediction(self, measurements_array, timestamp):
        predictor = self.predictor
        node_id = self.node_id
        base_station = self.base_station_gateway

        predictor.add_measurement(timestamp, measurements_array)
        if not predictor.in_prediction_horizon(timestamp):
            latest_measurements = predictor.get_measurements_in_current_prediction_horizon(timestamp)
            base_station.send_update(node_id, timestamp, latest_measurements)
            predictor.update_prediction_horizon(timestamp)
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
