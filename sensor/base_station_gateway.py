import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests import RequestException

from common import ThresholdMetric, ModelMetadata, LiteModel, Predictor, DataStorage


class BaseStationGateway:

    def __init__(self, node_id: str, base_address: str):
        """
        A façade class that manages all exchanges with the Base Station.

        Args:
            node_id: the unique ID of the sensor node
            base_address: The HTTP url of the Base Station, e.g., "192.168.0.1:8080".
        """
        logging.debug(f"Initializing BaseStationGateway with base address: {base_address}")
        self.base_address = base_address
        self.node_id = node_id
        response = requests.get(f'{self.base_address}/ping')
        if not response.ok:
            raise RequestException(
                f'Base Station is unreachable. GET {self.base_address}/ping returned {response.status_code}'
            )

    def register_node(self, threshold_metric: ThresholdMetric) -> requests.Response:
        """Registers the sensor node with the base station by providing its ID and threshold metric.

        Args:
            threshold_metric: The metric used to determine if a threshold has been reached.

        Returns:
            requests.Response: The response from the base station containing metadata and initial data in the body.

        Raises:
            requests.exceptions.RequestException: If an error occurs while sending the request.
        """
        body = {'threshold_metric': threshold_metric.to_dict()}
        logging.debug(f'Registering node {self.node_id} with base station at {self.base_address}: {body}')
        try:
            response = requests.post(f'{self.base_address}/register/{self.node_id}', json=body)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logging.error(f'Registration failed: {e}')
            raise

    def send_violation(self,
                       dt: datetime.datetime,
                       measurement: np.ndarray,
                       data: pd.DataFrame,
                       predictor: Predictor
                       ) -> Optional[Predictor]:
        """
        Sends a violation message to the base station, containing the timestamp of the violation, the measurement that
        triggered it, and the data required for updating the prediction horizon.
        """
        body = {
            'timestamp': dt.isoformat(),
            'measurement': list(measurement),
            'data': data.to_json(),
        }
        logging.debug(f'Node {self.node_id} handling violation by sending: {body}')

        try:
            response = requests.post(f'{self.base_address}/violation/{self.node_id}', json=body)
            response.raise_for_status()

            body = response.json()

            model_metadata = body.get('model_metadata')
            if model_metadata is not None:
                model_metadata = ModelMetadata.from_dict(model_metadata)
                model = self.fetch_model(model_metadata)

                new_predictor = Predictor(model, predictor.data)
                new_predictor.update_prediction_horizon(dt)
                return new_predictor

        except requests.exceptions.RequestException as e:
            logging.error(f'Error sending horizon update: {e}')
            raise e

    def fetch_model(self, model_metadata: ModelMetadata) -> LiteModel:
        """Fetches the prediction model from the base station.

        Args:
            model_metadata: The metadata of the model to fetch.

        Returns:
            The prediction model loaded from a TensorFlow Lite file.

        Raises:
            requests.exceptions.RequestException: If an error occurs while sending the request or loading the model.
        """
        try:
            r = requests.get(f'{self.base_address}/models/{self.node_id}')
            file_name = f'{model_metadata.uuid}.tflite'
            open(file_name, 'wb').write(r.content)
            return LiteModel.from_tflite_file(file_name, model_metadata)
        except requests.exceptions.RequestException as e:
            logging.error(f'Failed to fetch prediction model from base station {self.base_address}: {e}')
            raise

    def fetch_model_and_data(self, threshold_metric: ThresholdMetric) -> (LiteModel, DataStorage):
        """Fetches the prediction model and initial data from the base station.

        Args:
            threshold_metric: The metric used to determine if a threshold has been reached.

        Returns:
            A tuple with the prediction model loaded from a TensorFlow Lite file and the initial data used to train it.

        Raises:
            requests.exceptions.RequestException: If an error occurs while sending the request or loading the model.
        """
        try:
            # Register the node to receive model metadata and initial data
            response = self.register_node(threshold_metric)
            body = response.json()

            # Download the model file from the base station and load it into a LiteModel object
            metadata = ModelMetadata.from_dict(body.get('model_metadata'))
            model = self.fetch_model(metadata)

            # Load the initial data into a DataStorage object
            initial_df = pd.read_json(body.get('initial_df'))
            logging.debug(f'Node {self.node_id} fetched initial data for prediction model: {initial_df}')
            data = DataStorage(metadata.input_features, metadata.output_features)
            data.add_measurement_df(initial_df)

            return model, data

        except (requests.exceptions.RequestException, ValueError) as e:
            logging.error(f'Failed to fetch prediction model and data from base station: {e}')
            raise

    def send_update(self,
                    dt: datetime.datetime,
                    data: pd.DataFrame,
                    predictor: Predictor,
                    ) -> Optional[Predictor]:
        """Communicates a horizon update to the specified base address.

        Args:
            dt: The timestamp at which the horizon udpate is necessary a datetime object.
            data: The (redcued) measurements that occured in the current prediction horizon as a NumPy array.
            predictor: The Predictor currently used by the sensor.

        Raises:
            requests.exceptions.RequestException: An error occurred while sending the horizon update.

        """
        body = {
            'timestamp': dt.isoformat(),
            'data': data.to_json(),
        }
        logging.debug(f'Node {self.node_id} sending horizon update: {body}')

        try:
            response = requests.post(f'{self.base_address}/update/{self.node_id}', json=body)
            response.raise_for_status()

            body = response.json()
            model_metadata = body.get('model_metadata')
            if model_metadata is not None:
                model_metadata = ModelMetadata.from_dict(model_metadata)
                model = self.fetch_model(model_metadata)

                new_predictor = Predictor(model, predictor.data)
                new_predictor.update_prediction_horizon(dt)
                return new_predictor

        except requests.exceptions.RequestException as e:
            logging.error(f'Error sending horizon update: {e}')
            raise e

    def send_measurement(self, dt: datetime.datetime, measurement: np.ndarray):
        """Sends a single measurement to the specified base address.

        Args:
            dt: The measurement's timestamp as a datetime object.
            measurement: The measurement as a NumPy array.

        Raises:
            requests.exceptions.RequestException: An error occurred while sending the measurement.

        """
        body = {
            'timestamp': dt.isoformat(),
            'measurement': list(measurement),
        }
        logging.debug(f'Node {self.node_id} sending measurement: {body}')
        try:
            response = requests.post(f'{self.base_address}/measurement/{self.node_id}', json=body)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logging.error(f'Error sending measurement: {e}')
            raise e
