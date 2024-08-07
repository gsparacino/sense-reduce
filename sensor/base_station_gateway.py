import datetime
import logging
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd
import requests
from requests import RequestException, Response

from common import DataStorage, ModelMetadata


class NodeInitialization:
    def __init__(self, node_id: str, current_model: ModelMetadata, portfolio: list[str]):
        self.node_id: str = node_id
        self.current_model: ModelMetadata = current_model
        self.portfolio: list[str] = portfolio


class BaseStationGateway(ABC):

    @abstractmethod
    def register_node(self,
                      node_id: str,
                      threshold_metric: dict,
                      ) -> NodeInitialization:
        """
        Registers the sensor node with the base station by providing its ID and threshold metric. The Base Station may
        respond with the metadata of the model to fetch.

        :param node_id: ID of the sensor node, as a string.
        :param threshold_metric: The metric used to determine if a threshold has been reached, as a dict.

        :return: A tuple with: an instance of common.ModelMetadata containing the model's metadata; an instance of
        common.DataStorage containing the initial data for the model.
        """
        pass

    @abstractmethod
    def send_measurement(self, node_id: str, dt: datetime.datetime, measurement: np.ndarray) -> None:
        """
        Sends a single measurement to the Base Station.

        :param node_id: The node's unique ID.
        :param dt: The measurement's timestamp, as a datetime object.
        :param measurement: The measurement data, as a NumPy array.
        """
        pass

    @abstractmethod
    def get_model(self, node_id: str, model_id: str) -> bytes:
        """Fetches a specific prediction model from the Base Station.

        :param node_id: ID of the sensor node, as a string.
        :param model_id: ID of the model to fetch, as a string.

        :return: The bytes containing the prediction model.
        """
        pass

    @abstractmethod
    def get_model_metadata(self, node_id: str, model_id: str) -> ModelMetadata:
        """Fetches a specific prediction model's metadata from the Base Station.

        :param node_id: ID of the sensor node, as a string.
        :param model_id: ID of the model to fetch, as a string.

        :return: The ModelMetadata of the model.
        """
        pass

    @abstractmethod
    def synchronize(self,
                    node_id: str,
                    dt: datetime.datetime,
                    model_id: str,
                    measurements: pd.DataFrame
                    ) -> Optional[list[str]]:
        """
        Synchronizes with the Base Station, sending the latest measurements (if appropriate)and fetching the list of
        models available for the sensor.

        :param node_id: The node's unique ID.
        :param dt: The timestamp of the synchronization, i.e. the timestamp at which the latest measurement was read,
        as a datetime object.
        :param model_id: the ID of the model currently active on the node, as a string.
        :param measurements: The (reduced) measurements that occurred in the current prediction horizon, as a NumPy array.

        :return: A list of common.ModelMetadata, containing the metadata of the models that the sensor could fetch from
        the base station.
        """
        pass

    @abstractmethod
    def send_violation(self, node_id: str,
                       dt: datetime.datetime,
                       measurements: pd.DataFrame,
                       model_id: str,
                       portfolio: list[str],
                       request_new_model: bool = False) -> list[str]:
        """
        Notifies a violation on the Base Station.

        :param node_id: the node's unique ID
        :param dt: the timestamp of the violation event, as a datetime object.
        :param measurements: the measurements that caused the violation, as a NumPy array.
        :param model_id: the ID of the model currently active on the node, as a string.
        :param portfolio: the list of models currently stored in the node.
        :param request_new_model: If true, the node will ask the base station to send a new model on the next update
        """
        pass


class HttpBaseStationGateway(BaseStationGateway):
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
                      ) -> NodeInitialization:
        body = {
            'threshold_metric': threshold_metric,
            'node_id': node_id
        }
        logging.debug(f'Registering node {node_id} with base station at {self.base_address}: {body}')
        response = requests.post(f'{self.base_address}/nodes', json=body)
        if not response.ok:
            raise RequestException(f'POST {self.base_address}/nodes returned {response.status_code}')

        model_metadata = self._extract_model_metadata(response)
        portfolio = self._extract_models_portfolio(response)

        return NodeInitialization(node_id, model_metadata, portfolio)

    def send_measurement(self, node_id: str, dt: datetime.datetime, measurement: np.ndarray) -> None:
        body = {
            'timestamp': dt.isoformat(),
            'measurement': list(measurement),
        }
        logging.debug(f'Node {node_id} sending measurement: {body}')
        response = requests.post(f'{self.base_address}/measurement/{node_id}', json=body)
        if not response.ok:
            raise RequestException(f'POST {self.base_address}/measurement/{node_id} returned {response.status_code}')

    def get_model(self, node_id: str, model_id: str) -> bytes:
        logging.debug(f'Fetching model {model_id} from Base Station')
        r = requests.get(f'{self.base_address}/nodes/{node_id}/models/{model_id}')
        if not r.ok:
            raise RequestException(
                f'GET {self.base_address}/nodes/{node_id}/models/{model_id} returned {r.status_code}'
            )

        return r.content

    def get_model_metadata(self, node_id: str, model_id: str) -> ModelMetadata:
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
                    model_id: str,
                    measurements: pd.DataFrame
                    ) -> Optional[list[str]]:
        body = {
            'timestamp': dt.isoformat(),
            'measurements': measurements.to_json(),
            'model': model_id
        }
        logging.debug(f'Synchronization event: {body}')

        response = requests.post(f'{self.base_address}/nodes/{node_id}/sync', json=body)
        if not response.ok:
            raise RequestException(f'POST {self.base_address}/nodes/{node_id}/sync  returned {response.status_code}')

        return self._extract_models_portfolio(response)

    def send_violation(self, node_id: str,
                       dt: datetime.datetime,
                       measurements: pd.DataFrame,
                       model_id: str,
                       portfolio: list[str],
                       request_new_model: bool = False) -> list[str]:
        body = {
            'timestamp': dt.isoformat(),
            'measurements': measurements.to_json(),
            'model': model_id,
            'portfolio': portfolio,
            'needs_new_model': request_new_model
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
