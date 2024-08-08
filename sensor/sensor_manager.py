import datetime
import logging
import time
from typing import Optional

import pandas as pd

from common import Predictor, DataStorage
from sensor.abstract_sensor import AbstractSensor
from sensor.base_station_gateway import BaseStationGateway, NodeInitialization
from sensor.model_manager import ModelManager
from sensor.sensor_adaptive_strategy import SensorNodeAdaptiveStrategy
from sensor.violation import Violation

NodeID = str


class SensorManager:

    def __init__(self,
                 sensor: AbstractSensor,
                 base_station: BaseStationGateway,
                 model_manager: ModelManager,
                 adaptive_strategy: SensorNodeAdaptiveStrategy,
                 node_initialization: NodeInitialization,
                 prediction_interval: float,
                 ) -> None:
        """
        Manages a sensor's measurements and prediction models, as well as handling coordination with the Base Station.

        :param sensor: An implementation of AbstractSensor that provides measurements.
        :param base_station: An instance of BaseStationGateway that handles coordination with the Base Station.
        :param model_manager: An instance of ModelManager that manages the local portfolio of models.
        :param adaptive_strategy: An instance of AdaptiveStrategy that implements the adaptation logic of the sensor.
        :param node_initialization: An instance of NodeInitialization with the node's initial configuration.
        :param prediction_interval: The time interval between predictions.
        """
        logging.debug(f"Initializing SensorManager")

        self.node_id = node_initialization.node_id
        self.sensor: AbstractSensor = sensor
        self.model_manager: ModelManager = model_manager
        self.base_station: BaseStationGateway = base_station
        self.adaptive_strategy = adaptive_strategy
        self.node_initialization: NodeInitialization = node_initialization
        self._initial_model = node_initialization.current_model
        self._data_storage: DataStorage = self._init_data_storage(node_initialization)
        self.prediction_interval: float = prediction_interval
        self.predictor: Optional[Predictor] = None

    def run(self, time_interval: float, shutdown_dt: datetime.datetime = None) -> None:
        """Starts monitoring the sensor, coordinating with the Base Station by sending data notifying violations.
        """
        assert time_interval > 0, "time_interval must be greater than 0."

        node_id = self.node_id

        logging.debug(f"Sensor {node_id} started @ {datetime.datetime.now()}")

        while datetime.datetime.now() < shutdown_dt if shutdown_dt is not None else True:
            measurement, timestamp = self._get_measurement()
            logging.debug(f"Measurement @ {timestamp}: {measurement.values}")

            self.adaptive_strategy.execute(measurement, timestamp)

            # measurements_array = measurement.to_numpy()
            # self._data_storage.add_measurement(timestamp, measurements_array)
            #
            # if self._can_make_predictions():
            #     # if self.predictor is None:
            #     # self.predictor = self._init_predictor(self.prediction_interval,
            #     #                                       self.node_initialization,
            #     #                                       self._data_storage)
            #     if self.predictor is None:
            #         violation = Violation(node_id, timestamp, None, measurements_array, prediction_array)
            #         self.predictor = self.adaptive_strategy.handle_violation(violation)
            #
            #     predictor = self.predictor
            #     if not predictor.in_prediction_horizon(timestamp):
            #         self._refresh_expired_horizon(timestamp)
            #     prediction = predictor.get_prediction_at(timestamp)
            #     logging.debug(f"Prediction by {predictor.model_id} @ {timestamp}: {prediction.values}")
            #
            #     prediction_array = prediction.to_numpy()
            #     predictor.log_prediction(timestamp, prediction_array)
            #
            #     if self.adaptive_strategy.is_violation(measurements_array, prediction_array):
            #         violation = Violation(node_id, timestamp, predictor, measurements_array, prediction_array)
            #         self.predictor.add_violation(timestamp)
            #         self.predictor = self.adaptive_strategy.handle_violation(violation)
            # else:
            #     self.base_station.send_measurement(node_id, timestamp, measurement)

            time.sleep(time_interval)

        logging.debug(f"Sensor {node_id} stopped @ {datetime.datetime.now()}")

    def _can_make_predictions(self):
        return len(self._data_storage.get_measurements()) >= self._initial_model.input_length

    def _refresh_expired_horizon(self, timestamp):
        measurements = self.predictor.get_reduced_measurements_in_current_prediction_horizon(timestamp)
        self.base_station.synchronize(self.node_id, timestamp, self.predictor.model_id, measurements)
        self.predictor.add_prediction_horizon_update(timestamp)
        self.predictor.update_prediction_horizon(timestamp)
        logging.debug(f"Updated Prediction Horizon (scheduled update): \n {self.predictor.get_prediction_horizon()}")

    def _get_prediction(self, timestamp: datetime.datetime):
        predictor = self.predictor
        prediction = predictor.get_prediction_at(timestamp)
        return prediction

    def _get_measurement(self) -> (pd.Series, datetime):
        now = datetime.datetime.now()
        measurement = self.sensor.measurement
        while measurement is None:
            now = datetime.datetime.now()
            logging.warning(f'{now} - Failed reading measurement from sensor. Retrying...')
            measurement = self.sensor.measurement
        return measurement, now

    def _init_predictor(self, prediction_interval: float,
                        node_initialization: NodeInitialization,
                        data_storage: DataStorage) -> Predictor:
        self.model_manager.synchronize_models(node_initialization.portfolio)
        current_model = self.model_manager.get_model(node_initialization.current_model)
        period = datetime.timedelta(seconds=prediction_interval)
        predictor = Predictor(current_model, period, data_storage)
        predictor.update_prediction_horizon(datetime.datetime.now())
        return predictor

    def _init_data_storage(self, node_initialization: NodeInitialization) -> DataStorage:
        initial_df = node_initialization.initial_df
        if initial_df is None:
            return DataStorage(self._initial_model.input_features,
                               self._initial_model.output_features)
        else:
            return initial_df
