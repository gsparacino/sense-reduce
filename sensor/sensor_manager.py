import datetime
import logging
import os
import time
from enum import Enum, auto

from abstract_sensor import AbstractSensor
from common import ThresholdMetric, Predictor, LiteModel
from sensor.base_station_gateway import BaseStationGateway


class SensorManager:
    class _OperatingMode(Enum):
        NORMAL = auto(),
        ANOMALY = auto(),

    def __init__(self,
                 node_id: str,
                 sensor: AbstractSensor,
                 predictor: Predictor,
                 base_station_gateway: BaseStationGateway) -> None:
        """
        Manages a sensor measurements and prediction models, as well as handling coordination with the Base Station.

        :param node_id: The node id of the sensor node
        :param sensor: An implementation of AbstractSensor that provides measurements.
        :param predictor: An instance of Predictor that takes measurements as inputs and provides predictions.
        :param base_station_gateway: An instance of BaseStationGateway that handles coordination with the Base Station.
        """
        logging.debug(f"Initializing SensorManager with for node: {node_id}")
        self.node_id = node_id
        self.sensor = sensor
        self.predictor = predictor
        self.base_station_gateway = base_station_gateway
        self._operating_mode = self._OperatingMode.NORMAL

    def run(self, threshold_metric: ThresholdMetric, time_interval: float) -> None:
        """Starts monitoring the sensor, coordinating with the Base Station by sending data notifying violations.

        :param threshold_metric: A metric implementation that defines a threshold violation.
        :param time_interval: A float that represents the time period between consecutive measurements, in seconds.
        """
        assert time_interval > 0, "time_interval must be greater than 0."
        base_station = self.base_station_gateway
        predictor = self.predictor
        node_id = self.node_id

        while True:
            now = datetime.datetime.now()
            measurement = self.sensor.measurement

            while measurement is None:
                now = datetime.datetime.now()
                logging.warning(f'{now} - Failed reading measurement from sensor. Retrying...')
                measurement = self.sensor.measurement

            logging.debug(f"Measurement @ {now}: {measurement.values}")
            measurements_array = measurement.to_numpy()
            predictor.add_measurement(now, measurements_array)

            if not predictor.in_prediction_horizon(now):
                predictor.update_prediction_horizon(now)
                new_data = predictor.get_measurements_in_current_prediction_horizon(now)
                base_station.send_update(node_id, now, new_data)

            prediction = predictor.get_prediction_at(now)
            logging.debug(f"Prediction @ {now}: {prediction.values}")
            predictor.add_prediction(now, prediction.to_numpy())

            if threshold_metric.is_threshold_violation(measurement, prediction.to_numpy()):
                logging.info(f"Threshold violation: Measurement={measurement.values}, Prediction={prediction.values}")
                new_data = predictor.get_measurements_in_current_prediction_horizon(now)
                new_model = base_station.send_violation(node_id, now, measurements_array, new_data)
                if new_model is not None:
                    logging.debug(f"Base Station provided new model {new_model.uuid}.")
                    model_bytes = base_station.fetch_model(node_id, new_model)
                    basedir = os.path.abspath(os.path.dirname(__file__))
                    file_name = f'{new_model.uuid}.tflite'
                    model_path = os.path.join(basedir, 'models', file_name)
                    open(model_path, 'wb').write(model_bytes)
                    model = LiteModel.from_tflite_file(model_path, new_model)
                    new_predictor = Predictor(model, predictor.data, predictor.get_prediction_timedelta())
                    self.predictor = new_predictor
                    predictor = new_predictor
                else:
                    logging.debug(f"Base Station did not provide a new model. Switching to ANOMALY operating mode.")
                    self._operating_mode = self._OperatingMode.ANOMALY
                predictor.update_prediction_horizon(now)
                predictor.adjust_to_measurement(now, measurements_array, predictor.get_prediction_at(now).to_numpy())

            time.sleep(time_interval)
