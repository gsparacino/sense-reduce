import datetime
import logging
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests import RequestException, Response

from common import DataStorage, ModelMetadata


class NodeInitialization:
    def __init__(self, current_model: ModelMetadata, data_storage: DataStorage, portfolio: list[str]):
        self.current_model: ModelMetadata = current_model
        self.data_storage: DataStorage = data_storage
        self.portfolio: list[str] = portfolio


class BaseStationGateway:
    def __init__(self, base_address: str):
        """
        A façade class that abstracts away all communications with the Base Station.

        :param base_address: The HTTP url of the Base Station.
        """
        logging.debug(f"Initializing BaseStationGateway with base address: {base_address}")
        self.base_address = base_address
        response = requests.get(f'{self.base_address}/ping')
        if not response.ok:
            raise RequestException(
                f'Base Station is unreachable. GET {self.base_address}/ping returned {response.status_code}'
            )

    def register_node(self,
                      node_id: str,
                      threshold_metric: dict,
                      frequency: float) -> NodeInitialization:
        """
        Registers the sensor node with the base station by providing its ID and threshold metric. The Base Station may
        respond with the metadata of the model to fetch.

        :param node_id: ID of the sensor node, as a string.
        :param threshold_metric: The metric used to determine if a threshold has been reached, as a dict.
        :param frequency: The interval in seconds between consecutive measurements, as a float.

        :return: A tuple with: an instance of common.ModelMetadata containing the model's metadata; an instance of
        common.DataStorage containing the initial data for the model.

        :raises requests.RequestException: An error occurred while registering the node with the Base Station.
        """
        body = {
            'threshold_metric': threshold_metric,
            'prediction_period_s': frequency,
            'node_id': node_id
        }
        logging.debug(f'Registering node {node_id} with base station at {self.base_address}: {body}')
        response = requests.post(f'{self.base_address}/nodes', json=body)
        if not response.ok:
            raise RequestException(f'POST {self.base_address}/nodes returned {response.status_code}')

        model_metadata = self._extract_model_metadata(response)
        initial_df = self._extract_initial_df(response, model_metadata)
        portfolio = self._extract_models_portfolio(response)

        return NodeInitialization(model_metadata, initial_df, portfolio)

    def send_measurement(self, node_id: str, dt: datetime.datetime, measurement: np.ndarray) -> None:
        """
        Sends a single measurement to the Base Station.

        :param node_id: The node's unique ID.
        :param dt: The measurement's timestamp, as a datetime object.
        :param measurement: The measurement data, as a NumPy array.

        :raises requests.RequestException: An error occurred while sending the measurement.
        """
        body = {
            'timestamp': dt.isoformat(),
            'measurement': list(measurement),
        }
        logging.debug(f'Node {node_id} sending measurement: {body}')
        response = requests.post(f'{self.base_address}/measurement/{node_id}', json=body)
        if not response.ok:
            raise RequestException(f'POST {self.base_address}/measurement/{node_id} returned {response.status_code}')

    # def send_violation(self,
    #                    node_id: str,
    #                    dt: datetime.datetime,
    #                    measurement: np.ndarray,
    #                    latest_data: pd.DataFrame,
    #                    ) -> Optional[ModelMetadata]:
    #     """
    #     Sends a violation message to the Base Station, containing the timestamp of the violation, the measurement that
    #     triggered it, and the data required for updating the prediction horizon.
    #
    #     :param node_id: The node's unique ID.
    #     :param dt: The measurement's timestamp, as a datetime object.
    #     :param measurement: The measurement that triggered the violation, as a NumPy array.
    #     :param latest_data: The latest measurements gathered by the node, as a pandas DataFrame.
    #
    #     :return: If the Base Station requires the node to switch model, a common.ModelMetadata containing the metadata
    #     of the new model; otherwise None is returned.
    #
    #     :raises requests.RequestException: An error occurred while sending the violation.
    #     """
    #     body = {
    #         'timestamp': dt.isoformat(),
    #         'measurement': list(measurement),
    #         'data': latest_data.to_json(),
    #     }
    #     logging.debug(f'Node {node_id} handling violation by sending: {body}')
    #
    #     response = requests.post(f'{self.base_address}/violation/{node_id}', json=body)
    #     if not response.ok:
    #         raise RequestException(f'POST {self.base_address}/violation/{node_id} returned {response.status_code}')
    #
    #     return self._extract_model_metadata(response)

    def fetch_model_bytes(self, node_id: str, model_id: str) -> bytes:
        """Fetches a specific prediction model from the Base Station.

        :param node_id: ID of the sensor node, as a string.
        :param model_id: ID of the model to fetch, as a string.

        :return: The bytes containing the prediction model.

        :raises requests.RequestException: An error occurred while fetching the model.
        """
        logging.debug(f'Fetching model {model_id} from Base Station')
        r = requests.get(f'{self.base_address}/nodes/{node_id}/models/{model_id}')
        if not r.ok:
            raise RequestException(
                f'GET {self.base_address}/nodes/{node_id}/models/{model_id} returned {r.status_code}'
            )

        return r.content

    def fetch_model_metadata(self, node_id: str, model_id: str) -> ModelMetadata:
        """Fetches a specific prediction model's metadata from the Base Station.

        :param node_id: ID of the sensor node, as a string.
        :param model_id: ID of the model to fetch, as a string.

        :return: The ModelMetadata of the model.

        :raises requests.RequestException: An error occurred while fetching the model.
        """
        logging.debug(f'Fetching model {model_id} metadata from Base Station')
        r = requests.get(f'{self.base_address}/nodes/{node_id}/models/{model_id}/metadata')
        if not r.ok:
            raise RequestException(
                f'GET {self.base_address}/nodes/{node_id}/models/{model_id}/metadata returned {r.status_code}'
            )

        metadata = ModelMetadata.from_dict(r.json())

        return metadata

    def synchronize(self,
                    node_id: str,
                    dt: datetime.datetime,
                    measurements: pd.DataFrame
                    ) -> Optional[list[str]]:
        """
        Synchronizes with the Base Station, sending the latest measurements (if appropriate)and fetching the list of
        models available for the sensor.

        :param node_id: The node's unique ID.
        :param dt: The timestamp of the synchronization, i.e. the timestamp at which the latest measurement was read,
        as a datetime object.
        :param measurements: The (reduced) measurements that occurred in the current prediction horizon, as a NumPy array.

        :return: A list of common.ModelMetadata, containing the metadata of the models that the sensor could fetch from
        the base station.

        :raises requests.RequestException: An error occurred while sending the request.
        """
        body = {
            'timestamp': dt.isoformat(),
            'measurements': measurements.to_json(),
        }
        logging.debug(f'Synchronization event: {body}')

        response = requests.post(f'{self.base_address}/nodes/{node_id}/sync', json=body)
        if not response.ok:
            raise RequestException(f'POST {self.base_address}/nodes/{node_id}/sync  returned {response.status_code}')

        return self._extract_models_portfolio(response)

    def request_new_model(self, node_id: str, dt: datetime.datetime, data: pd.DataFrame) -> Optional[ModelMetadata]:
        body = {
            'timestamp': dt.isoformat(),
            'data': data.to_json(),
        }
        logging.debug(f'Requesting new model: {body}')
        response = requests.post(f'{self.base_address}/nodes/{node_id}/models/new', json=body)
        if not response.ok:
            raise RequestException(
                f'POST {self.base_address}/nodes/{node_id}/models/new returned {response.status_code}'
            )

        return self._extract_model_metadata(response)

    def send_violation(self, node_id: str, dt: datetime.datetime, data: pd.DataFrame) -> list[str]:
        body = {
            'timestamp': dt.isoformat(),
            'measurements': data.to_json(),
        }
        logging.debug(f'Violation event: {body}')
        response = requests.post(f'{self.base_address}/nodes/{node_id}/violation', json=body)
        if not response.ok:
            raise RequestException(
                f'POST {self.base_address}/nodes/{node_id}/violation returned {response.status_code}'
            )

        return self._extract_models_portfolio(response)

    @staticmethod
    def _extract_model_metadata(response: Response) -> Optional[ModelMetadata]:
        """
        Maps a requests.Response object to a ModelMetadata object, if the relevant data is included in the response.

        :param response: The requests.Response object, with the Base Station response.
        :return: A ModelMetadata object if the response body includes a valid 'model_metadata' property, otherwise None.
        """
        body = response.json()
        new_model_metadata = body.get('model_metadata')
        if new_model_metadata is not None:
            return ModelMetadata.from_dict(new_model_metadata)
        else:
            return None

    @staticmethod
    def _extract_models_portfolio(response: Response) -> Optional[list[str]]:
        body = response.json()
        portfolio: list = body.get('portfolio')
        if portfolio is not None:
            return portfolio

        return None

    @staticmethod
    def _extract_initial_df(response: Response, metadata: ModelMetadata) -> Optional[DataStorage]:
        """
        Maps a requests.Response object to a DataStorage object containing the initial dataframe for the model, if the
        relevant data is included in the response.

        :param response: The requests.Response object, with the Base Station response.
        :param metadata: The ModelMetadata object containing the metadata of the model.
        :return: A DataStorage object if the response body includes a valid 'initial_df' property, otherwise None.
        """
        body = response.json()
        initial_df = body.get('initial_df')

        if initial_df is not None:
            data = DataStorage(metadata.input_features, metadata.output_features)
            df = pd.read_json(initial_df)
            data.add_measurement_df(df)
            return data
        else:
            return None
