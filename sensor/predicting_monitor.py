import datetime
import logging
import time

from abstract_sensor import AbstractSensor
from common.predictor import Predictor
from common.threshold_metric import ThresholdMetric
from sensor.base_station_gateway import BaseStationGateway


class PredictingMonitor:
    """Continuously watches a sensors' measurements and compares it to the prediction of the model.

    Args:
        sensor: An implementation of AbstractSensor this monitor should watch.
        predictor: A Predictor that provides the predicted measurements.
    """

    def __init__(self, sensor: AbstractSensor, predictor: Predictor) -> None:
        self.sensor = sensor
        self.predictor = predictor

    def monitor(self,
                node_id: str,
                threshold_metric: ThresholdMetric,
                interval_seconds: float,
                base_station_gateway: BaseStationGateway) -> None:
        """Starts monitoring the sensor and calls the callback_fn if a threshold violation occurs.

        Args:
            node_id: The unique id of the sensor node.
            threshold_metric: A metric implementation that defines a threshold violation.
            interval_seconds: Defines the regular checking interval in seconds.
            base_station_gateway: An instance of BaseStationGateway that handles all exchanges with the base station.
        """
        while True:
            measurement = self.sensor.measurement
            while measurement is None:
                logging.warning(f'{datetime.datetime.now()} - Failed reading measurement from sensor. Retrying...')
                measurement = self.sensor.measurement

            now = datetime.datetime.now()
            p = self.predictor
            p.add_measurement(now, measurement.to_numpy())

            try:
                prediction = p.get_prediction_at(now)
            except ValueError:
                # TODO: keep track of last synchronization and only send required data
                new_data = p.get_measurements_in_current_prediction_horizon(now)
                p.update_prediction_horizon(now)
                new_predictor = base_station_gateway.send_update(node_id, now, new_data, p)
                if new_predictor is not None:
                    self.predictor = new_predictor
                prediction = p.get_prediction_at(now)

            p.add_prediction(now, prediction.to_numpy())

            if threshold_metric.is_threshold_violation(measurement, prediction.to_numpy()):
                # TODO: keep track of last synchronization and only send required data
                new_data = p.get_measurements_in_current_prediction_horizon(now)
                new_predictor = base_station_gateway.send_violation(node_id, now, measurement.to_numpy(), new_data, p)
                if new_predictor is not None:
                    self.predictor = new_predictor
                p.update_prediction_horizon(now)
                p.adjust_to_measurement(now, measurement.to_numpy(), p.get_prediction_at(now).to_numpy())

            time.sleep(interval_seconds)
