import datetime
import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
import requests
from flask import Response
from requests import RequestException

from common.model_metadata import ModelMetadata
from common.sensor_adaptation_goal import SensorAdaptationGoal
from common.sensor_knowledge_update import SensorKnowledgeUpdate, SensorKnowledgeInitialization


class BaseStationGateway(ABC):

    @abstractmethod
    def register_node(self,
                      initial_df: Optional[pd.DataFrame],
                      input_features: list[str],
                      output_features: list[str],
                      ) -> SensorKnowledgeInitialization:
        pass

    @abstractmethod
    def send_violation(self,
                       dt: datetime.datetime,
                       configuration_id: str,
                       measurement: pd.Series,
                       data: pd.DataFrame
                       ) -> SensorKnowledgeUpdate:
        pass

    @abstractmethod
    def send_update(self,
                    dt: datetime.datetime,
                    data: pd.DataFrame,
                    configuration_id: str
                    ) -> SensorKnowledgeUpdate:
        pass

    @abstractmethod
    def send_measurement(self,
                         dt: datetime.datetime,
                         measurement: pd.Series,
                         configuration_id: str
                         ) -> SensorKnowledgeUpdate:
        pass

    @abstractmethod
    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        """Fetches a specific prediction model's metadata from the Base Station.

        :param model_id: ID of the model to fetch, as a string.

        :return: The ModelMetadata of the model.
        """
        pass

    @abstractmethod
    def sync(self, dt: datetime.datetime) -> SensorKnowledgeUpdate:
        """
        Synchronizes the sensor knowledge with the base station.

        Returns
            The sensor knowledge update received from the base station.
        """
        pass

    @abstractmethod
    def fetch_model(self, model_id: str) -> bytes:
        pass


class HttpBaseStationGateway(BaseStationGateway):

    def __init__(self, node_id: str, base_address: str):
        self.node_id = node_id
        self.base_address = base_address
        response = requests.get(f'{self.base_address}/ping')
        if not response.ok:
            raise RequestException(
                f'Base Station is unreachable or unhealthy. GET {self.base_address}/ping returned {response.status_code}'
            )

    def register_node(self,
                      initial_df: Optional[pd.DataFrame],
                      input_features: list[str],
                      output_features: list[str],
                      ) -> SensorKnowledgeInitialization:
        body = {
            'input_features': input_features,
            'output_features': output_features
        }

        if initial_df is not None:
            body['initial_df'] = initial_df.to_json()

        logging.debug(f'Registering node with base station at {self.base_address}')
        response = requests.post(f'{self.base_address}/nodes', json=body)
        if not response.ok:
            raise RequestException(f'POST {self.base_address}/nodes returned {response.status_code}')
        body = response.json()
        node_id = body['node_id']
        adaptation_goals = self._extract_adaptation_goals(response)
        model_metadata = self._extract_model_metadata(response)
        portfolio = self._extract_models_portfolio(response)
        # TODO: add ability to receive initial data from BS

        return SensorKnowledgeInitialization(
            node_id=node_id,
            adaptation_goals=adaptation_goals,
            base_model_metadata=model_metadata,
            models_portfolio=portfolio,
            initial_df=initial_df
        )

    def send_violation(self,
                       dt: datetime.datetime,
                       configuration_id: str,
                       measurement: pd.Series,
                       data: pd.DataFrame) -> SensorKnowledgeUpdate:
        body = {
            'timestamp': dt.isoformat(),
            'measurements': measurement.to_json(),
            'configuration_id': configuration_id,
            'data': data.to_json()
        }
        logging.debug(f'Violation event: {body}')
        response = requests.post(f'{self.base_address}/nodes/{self.node_id}/violation', json=body)
        if not response.ok:
            raise RequestException(
                f'POST {self.base_address}/nodes/{self.node_id}/violation returned {response.status_code}'
            )

        return self.to_knowledge_update(response)

    def send_update(self, dt: datetime.datetime, data: pd.DataFrame, configuration_id: str) -> SensorKnowledgeUpdate:
        body = {
            'timestamp': dt.isoformat(),
            'data': data.to_json(),
            'configuration_id': configuration_id
        }
        logging.debug(f'Horizon update event @ {dt.isoformat()}')

        response = requests.post(f'{self.base_address}/nodes/{self.node_id}/update', json=body)
        if not response.ok:
            raise RequestException(
                f'POST {self.base_address}/nodes/{self.node_id}/sync  returned {response.status_code}')

        return self.to_knowledge_update(response)

    def send_measurement(self,
                         dt: datetime.datetime,
                         measurement: pd.Series,
                         configuration_id: str) -> SensorKnowledgeUpdate:
        body = {
            'timestamp': dt.isoformat(),
            'measurement': measurement.to_json(),
            'configuration_id': configuration_id
        }
        logging.debug(f'Sending measurement @ {dt.isoformat()}')
        response = requests.post(f'{self.base_address}/nodes/{self.node_id}/measurement', json=body)
        if not response.ok:
            raise RequestException(
                f'POST {self.base_address}/nodes/{self.node_id}/measurement returned {response.status_code}')

        return self.to_knowledge_update(response)

    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        logging.debug(f'Fetching model {model_id} metadata from Base Station')
        r = requests.get(f'{self.base_address}/nodes/{self.node_id}/models/{model_id}/metadata')
        if not r.ok:
            raise RequestException(
                f'GET {self.base_address}/nodes/{self.node_id}/models/{model_id}/metadata returned {r.status_code}'
            )

        metadata = ModelMetadata.from_dict(r.json())

        return metadata

    def fetch_model(self, model_id: str) -> bytes:
        logging.debug(f'Fetching model {model_id} from Base Station')
        r = requests.get(f'{self.base_address}/nodes/{self.node_id}/models/{model_id}')
        if not r.ok:
            raise RequestException(
                f'GET {self.base_address}/nodes/{self.node_id}/models/{model_id} returned {r.status_code}'
            )

        return r.content

    def to_knowledge_update(self, response):
        """
        Converts a Base Station response into a KnowledgeUpdate object.

        Args:
            response: the BaseStation response.

        Returns: an instance of KnowledgeUpdate with the configuration updates provided by the BaseStation.
        """
        goals = self._extract_adaptation_goals(response)
        portfolio = self._extract_models_portfolio(response)
        return SensorKnowledgeUpdate(goals, portfolio)

    @staticmethod
    def _extract_model_metadata(response: Response) -> Optional[ModelMetadata]:
        """
        Maps a requests.Response object to a ModelMetadata object, if the relevant data is included in the response.

        :param response: The requests.Response object, with the Base Station response.
        :return: A ModelMetadata object if the response body includes a valid 'model_metadata' property, otherwise None.
        """
        body = response.json()
        new_model_metadata = body.get('base_model_metadata')
        if new_model_metadata is not None:
            return ModelMetadata.from_dict(new_model_metadata)
        else:
            return None

    @staticmethod
    def _extract_models_portfolio(response: Response) -> Optional[list[str]]:
        body = response.json()
        portfolio: list = body.get('models_portfolio')
        if portfolio is not None:
            return portfolio

        return None

    @staticmethod
    def _extract_adaptation_goals(response: Response) -> Optional[list[SensorAdaptationGoal]]:
        body = response.json()
        goals: list[dict] = body.get('adaptation_goals')
        if goals is not None:
            adaptation_goals: list[SensorAdaptationGoal] = []
            for goal in goals:
                adaptation_goal = SensorAdaptationGoal.from_dict(goal)
                adaptation_goals.append(adaptation_goal)
            return adaptation_goals
        return None
