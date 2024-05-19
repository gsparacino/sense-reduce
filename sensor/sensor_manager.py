import datetime
import logging
import os
import time
from enum import Enum, auto
from typing import Dict

import pandas as pd

from abstract_sensor import AbstractSensor
from common import ThresholdMetric, Predictor, LiteModel
from sensor.base_station_gateway import BaseStationGateway

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
                 mode: OperatingMode = OperatingMode.DATA_REDUCTION,
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
        self.current_predictor: Predictor = predictor
        self._predictors: Dict[str, Predictor] = {}
        self._add_predictor(predictor)
        self.base_station_gateway: BaseStationGateway = base_station_gateway
        self._operating_mode = mode

    def run(self, threshold_metric: ThresholdMetric, time_interval: float) -> None:
        """Starts monitoring the sensor, coordinating with the Base Station by sending data notifying violations.

        :param threshold_metric: A metric implementation that defines a threshold violation.
        :param time_interval: A float that represents the time period between consecutive measurements, in seconds.
        """
        assert time_interval > 0, "time_interval must be greater than 0."
        base_station = self.base_station_gateway
        predictor = self.current_predictor
        node_id = self.node_id

        while True:
            measurement, timestamp = self._get_measurement()
            logging.debug(f"Measurement @ {timestamp}: {measurement.values}")

            measurements_array = measurement.to_numpy()
            predictor.add_measurement(timestamp, measurements_array)

            if self._operating_mode == SensorManager.OperatingMode.DATA_REDUCTION:
                prediction = self._get_prediction(measurements_array, timestamp)
                logging.debug(f"Prediction @ {timestamp}: {prediction.values}")

                prediction_array = prediction.to_numpy()
                predictor.add_prediction(timestamp, prediction_array)

                if threshold_metric.is_threshold_violation(measurements_array, prediction_array):
                    # TODO: check if the violation actually requires a model change
                    logging.info(
                        f"Threshold violation: Measurement={measurement.values}, Prediction={prediction.values}"
                    )
                    # TODO: add local model selection logic
                    self._operating_mode = SensorManager.OperatingMode.MODEL_TRAINING
                    logging.debug(f"Switching to {self._operating_mode.name}")

            if self._operating_mode == SensorManager.OperatingMode.MODEL_TRAINING:
                new_data = predictor.get_measurements_in_current_prediction_horizon(timestamp)
                new_model_metadata = base_station.request_new_model(node_id, timestamp, new_data)
                if new_model_metadata is not None:
                    logging.debug(f"Base Station provided new model {new_model_metadata.model_id}")
                    model_bytes = base_station.fetch_model_file(node_id, new_model_metadata.model_id)
                    logging.debug(f"Model {new_model_metadata.model_id} fetched")
                    predictor = self._create_new_predictor(new_model_metadata, model_bytes)
                    self.current_predictor = predictor
                    self._operating_mode = SensorManager.OperatingMode.DATA_REDUCTION
                    logging.debug(f"Switching to {self._operating_mode.name}")
                predictor.update_prediction_horizon(timestamp)
                predictor.adjust_to_measurement(timestamp, measurements_array,
                                                predictor.get_prediction_at(timestamp).to_numpy())

            time.sleep(time_interval)

    def _create_new_predictor(self, metadata, model_bytes: bytes) -> Predictor:
        predictor = self.current_predictor

        basedir = os.path.abspath(os.path.dirname(__file__))
        file_name = f'{metadata.model_id}.tflite'
        model_path = os.path.join(basedir, 'models', file_name)
        open(model_path, 'wb').write(model_bytes)
        model = LiteModel.from_tflite_file(model_path, metadata)

        return Predictor(model, predictor.data, predictor.get_prediction_timedelta())

    def _get_prediction(self, measurements_array, timestamp):
        predictor = self.current_predictor
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

    def _add_predictor(self, predictor: Predictor) -> None:
        self._predictors[predictor.model_metadata.model_id] = predictor
