import datetime
import logging
import os
import time
import uuid

import requests

from common import Predictor, ThresholdMetric, L2Threshold
from predicting_monitor import PredictingMonitor
from sensor.base_station_gateway import BaseStationGateway

logging.basicConfig(level=logging.DEBUG)


def run(threshold_metric: ThresholdMetric,
        base_address: str,
        data_reduction_mode: str,
        wifi_toggle: bool,
        check_interval: float,
        ) -> None:
    """Starts a sensor node in SenseReduce, which connects to a base station and starts monitoring.

    Args:
        threshold_metric (ThresholdMetric): The metric used to determine if a threshold has been reached.
        base_address (str): The address of the base station, e.g., "192.168.0.1:100".
        data_reduction_mode (str): The data reduction mode applied, either "none" or "predict".
        wifi_toggle (bool): A flag indicating whether Wi-Fi should be turned off between transmissions.
        check_interval (float): The regular interval in seconds for checking the sensor's readings.

    Returns:
        None

    Raises:
        ValueError: If an invalid data reduction mode is specified.
    """

    logging.info(f'Starting sensor node with ID={NODE_ID} in "{data_reduction_mode}" mode, '
                 f'threshold={threshold_metric} and base={base_address}...'
                 )
    sensor = TemperatureSensor()
    base_station = BaseStationGateway(node_id=NODE_ID, base_address=base_address)

    if data_reduction_mode == 'none':
        base_station.register_node(threshold_metric)
        while True:
            current_time = datetime.datetime.now()
            base_station.send_measurement(current_time, sensor.measurement.values)
            time.sleep(check_interval)

    elif data_reduction_mode == 'predict':
        model, data = base_station.fetch_model_and_data(threshold_metric)
        predictor = Predictor(model, data)
        predictor.update_prediction_horizon(datetime.datetime.now())
        monitor = PredictingMonitor(sensor, predictor)

        monitor.monitor(
            NODE_ID,
            threshold_metric=threshold_metric,
            interval_seconds=check_interval,
            base_station_gateway=base_station
        )

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
    import argparse

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
    ARGS = parser.parse_args()

    if ARGS.sensor == 'ds18b20':
        from temperature_sensor_ds18b20 import TemperatureSensor
    elif ARGS.sensor == 'sense-hat':
        from temperature_sensor_sense_hat import TemperatureSensor
    elif ARGS.sensor == 'dht22':
        from temperature_sensor_dht22 import TemperatureSensor
    elif ARGS.sensor == 'mock':
        from temperature_sensor_mock import TemperatureSensor
    else:
        logging.error(f'Unsupported sensor type: {ARGS.sensor}. Aborting...')
        exit(1)

    NODE_ID = ARGS.id
    THRESHOLD = L2Threshold(ARGS.threshold, [0], [0])
    run(THRESHOLD, ARGS.base, ARGS.mode, ARGS.wifi, ARGS.interval)
