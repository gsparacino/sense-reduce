import argparse
import datetime
import logging
import os
import time
import uuid

import requests

from common import Predictor, ThresholdMetric, L2Threshold
from sensor.abstract_sensor import AbstractSensor
from sensor.base_station_gateway import BaseStationGateway
from sensor.model_manager import ModelManager
from sensor.sensor_manager import SensorManager

logging.basicConfig(level=logging.DEBUG)

model_manager = ModelManager('models')  # TODO: make the model's path configurable


def run(threshold_metric: ThresholdMetric,
        base_address: str,
        data_reduction_mode: str,
        wifi_toggle: bool,
        time_interval: float,
        sensor: AbstractSensor
        ) -> None:
    """Starts a sensor node in SenseReduce, which connects to a base station and starts monitoring.

    Args:
        threshold_metric (ThresholdMetric): The metric used to determine if a threshold has been reached.
        base_address (str): The address of the base station, e.g., "192.168.0.1:100".
        data_reduction_mode (str): The data reduction mode applied, either "none" or "predict".
        wifi_toggle (bool): A flag indicating whether Wi-Fi should be turned off between transmissions.
        time_interval (float): The regular interval in seconds for checking the sensor's readings.
        sensor (AbstractSensor): The implementation of AbstractSensor from which measurements are collected

    Returns:
        None

    Raises:
        ValueError: If an invalid data reduction mode is specified.
    """

    logging.info(f'Starting sensor node with ID={NODE_ID} in "{data_reduction_mode}" mode, '
                 f'threshold={threshold_metric} and base={base_address}...'
                 )
    prediction_period = time_interval * 3  # TODO: make prediction_period configurable
    base_station = BaseStationGateway(base_address)
    threshold_metric_to_dict = threshold_metric.to_dict()

    if data_reduction_mode == 'none':
        # TODO: use NodeManager anyway
        base_station.register_node(NODE_ID, threshold_metric_to_dict, prediction_period)
        while True:
            current_time = datetime.datetime.now()
            base_station.send_measurement(NODE_ID, current_time, sensor.measurement.values)
            time.sleep(time_interval)

    elif data_reduction_mode == 'predict':
        model_id, model_metadata, data_storage = (
            base_station.register_node(NODE_ID, threshold_metric_to_dict, prediction_period))

        model_bytes: bytes = base_station.fetch_model_file(NODE_ID, model_id)
        period = datetime.timedelta(seconds=prediction_period)
        model = model_manager.save_model(model_id, model_bytes, model_metadata)

        predictor = Predictor(model, data_storage, period)
        predictor.update_prediction_horizon(datetime.datetime.now())

        sensor_manager = SensorManager(NODE_ID, sensor, predictor, base_station,
                                       SensorManager.OperatingMode.DATA_REDUCTION)
        sensor_manager.run(threshold_metric, time_interval)
    else:
        raise ValueError(f'Unsupported data reduction mode: {data_reduction_mode}')


def wifi_wrapper(wifi_toggle: bool, func, *args, **kwargs):
    """
    Wrapper function that handles toggling Wi-Fi before and after a function call.
    If `wifi_toggle` is True, Wi-Fi is turned on before the function call and off after it.
    The function call is executed regardless of the `wifi_toggle` setting.
    """
    if wifi_toggle:
        os.system('sudo rfkill unblock wifi')
        # wait for Wi-Fi to connect
        base_url = kwargs.get('base', 'http://192.168.8.110:5000')
        wait_for_wifi(base_url)
    func(*args, **kwargs)
    if wifi_toggle:
        os.system('sudo rfkill block wifi')


def wait_for_wifi(base_url: str, timeout: int = 30):
    """
    Waits for the Wi-Fi to connect to the base station.
    """
    start_time = time.monotonic()
    while True:
        try:
            r = requests.get(f'{base_url}/ping')
            if r.ok:
                end_time = time.monotonic()
                logging.debug(f'Connected to {base_url} in {end_time - start_time} seconds')
                return
        except requests.exceptions.RequestException:
            pass
        elapsed_time = time.monotonic() - start_time
        if elapsed_time >= timeout:
            logging.warning(f'Timed out waiting for Wi-Fi to connect to {base_url}')
            return
        time.sleep(0.5)  # wait 0.5 second before retrying


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Start a new sensor node.')
    parser.add_argument('sensor', type=str, choices=['ds18b20', 'sense-hat', 'dht22', 'mock'],
                        help='The sensor type to use for measurements.'
                        )
    parser.add_argument('base', type=str,
                        help='The address of the base station, e.g., 192.168.0.1:100.'
                        )
    parser.add_argument('--mode', type=str, choices=['none', 'predict'], default='predict',
                        help='The operation mode for data reduction, either "none" or "predict", default: "predict"'
                        )
    parser.add_argument('--wifi', action='store_true',
                        help='A flag for turning off Wi-Fi in-between transmissions.'
                        )
    parser.add_argument('--interval', type=float, default=5.0,
                        help='The regular monitoring interval for measurements in seconds.'
                        )
    parser.add_argument('--threshold', type=float, default=1.0,
                        help='The threshold in degrees Celsius above which to report to the base station, default: 1.0.'
                        )
    parser.add_argument('--id', type=str, default=uuid.uuid1(),
                        help='The unique ID of the node. If multiple nodes have the same ID, behavior is undefined.'
                             'If not provided, a UUID is generated with uuid.uuid1().'
                        )
    parser.add_argument('--csv', type=str,
                        help='The path of a .csv file containing pre-defined data to be returned by the mock sensor.'
                             'If no csv file path is provided, the mock sensor will generate random data.'
                             'This argument is ignored if the value of the "sensor" argument is not "mock".'
                        )
    ARGS = parser.parse_args()

    if ARGS.sensor == 'ds18b20':
        from temperature_sensor_ds18b20 import DS18B20Sensor

        sensor = DS18B20Sensor()
    elif ARGS.sensor == 'sense-hat':
        from temperature_sensor_sense_hat import HatSensor

        sensor = HatSensor()
    elif ARGS.sensor == 'dht22':
        from temperature_sensor_dht22 import DHT22Sensor

        sensor = DHT22Sensor()
    elif ARGS.sensor == 'mock':
        if ARGS.csv is not None:
            from multivariate_sensor_mock_csv import MultivariateCsvMockSensor

            sensor = MultivariateCsvMockSensor(ARGS.csv)
        else:
            from temperature_sensor_mock_random import RandomMockSensor

            sensor = RandomMockSensor()
    else:
        logging.error(f'Unsupported sensor type: {ARGS.sensor}. Aborting...')
        exit(1)

    NODE_ID = ARGS.id
    THRESHOLD = L2Threshold(ARGS.threshold, [0], [0])
    run(THRESHOLD, ARGS.base, ARGS.mode, ARGS.wifi, ARGS.interval, sensor)
