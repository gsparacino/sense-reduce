import logging
import os.path
import uuid

import pandas as pd

from common.threshold_metric import L2Threshold
from sensor.base_station_gateway import HttpBaseStationGateway
from sensor.multivariate_sensor_mock import MultivariateSensorMock
from sensor.sensor_analyzer import PortfolioSensorAnalyzer
from sensor.sensor_executor import SequentialSensorExecutor
from sensor.sensor_knowledge import SensorKnowledge
from sensor.sensor_manager import SensorManager
from sensor.sensor_monitor import MultivariateSensorMonitor
from sensor.sensor_planner import PortfolioSensorPlanner

logging.basicConfig(level=logging.DEBUG)
base_dir = os.path.abspath(os.path.dirname(__file__))


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
    # TODO: use configurations
    temp_node_id = str(uuid.uuid4())
    input_features = ["TL", "P", "RF", "SO"]
    output_features = ['TL']

    initial_df_path = os.path.join(base_dir, 'data', 'zamg_vienna_hourly.pickle')
    vienna_df = pd.read_pickle(str(initial_df_path))
    vienna_df = vienna_df[input_features]
    vienna_df.index = pd.to_datetime(vienna_df.index)
    initial_df = vienna_df[vienna_df.index.year == 2019]

    base_station_gateway = HttpBaseStationGateway(temp_node_id, base_address)
    knowledge_initialization = base_station_gateway.register_node(initial_df, input_features, output_features)

    knowledge = SensorKnowledge.from_initialization(base_station_gateway, initial_df, knowledge_initialization)
    # TODO: make MAPEK components configurable
    executor = SequentialSensorExecutor
    planner = PortfolioSensorPlanner
    analyzer = PortfolioSensorAnalyzer
    monitor = MultivariateSensorMonitor
    sensor_data = vienna_df[vienna_df.index.year >= 2020]
    sensor = MultivariateSensorMock(sensor_data)
    model_dir = os.path.join(base_dir, 'models')
    sensor_manager = SensorManager(
        model_dir, input_features, output_features, sensor, initial_df, base_station_gateway,
        monitor, analyzer, planner, executor, knowledge
    )
    sensor_manager.run()


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
        pass
    elif ARGS.sensor == 'sense-hat':
        pass
    elif ARGS.sensor == 'dht22':
        pass
    elif ARGS.sensor == 'mock':
        pass
    else:
        logging.error(f'Unsupported sensor type: {ARGS.sensor}. Aborting...')
        exit(1)

    NODE_ID = ARGS.id
    THRESHOLD = L2Threshold(ARGS.threshold, [0], [0])
    run(THRESHOLD, ARGS.base, ARGS.mode, ARGS.wifi, ARGS.interval)
