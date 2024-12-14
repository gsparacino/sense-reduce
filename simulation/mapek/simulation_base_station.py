import datetime
import json
import os
import uuid
from typing import Optional

import pandas as pd

from base.cluster_manager import ClusterManager
from common.model_metadata import ModelMetadata
from common.model_utils import load_model_from_savemodel, to_tflite_model_bytes
from common.resource_profiler import Profiler
from common.sensor_knowledge_update import SensorKnowledgeUpdate, SensorKnowledgeInitialization
from sensor.base_station_gateway import BaseStationGateway


def _serialize_df(data: pd.DataFrame):
    if data is not None:
        serialized_data = data.copy()
        serialized_data.index = serialized_data.index.map(lambda i: i.isoformat())
        serialized_data = serialized_data.to_dict(orient='index')
    else:
        serialized_data = {}
    return serialized_data


def _estimate_payload_size(data: dict):
    json_payload = json.dumps(data)
    return len(json_payload.encode('utf-8'))


def _fetch_model(model_dir, model_id, profiler):
    model = load_model_from_savemodel(os.path.join(model_dir, model_id))
    model_bytes = to_tflite_model_bytes(model.model)
    if profiler is not None:
        profiler.add_log_entry(
            datetime.datetime.now(),
            0,
            "fetch_model",
            model_id,
            0,
            0,
            len(model_bytes),
            0
        )
    return model_bytes


def _get_model_metadata(model_dir, model_id, profiler):
    model = load_model_from_savemodel(os.path.join(model_dir, model_id))
    if profiler is not None:
        response_size = _estimate_payload_size(model.metadata.to_dict())
        profiler.add_log_entry(
            datetime.datetime.now(),
            0,
            "get_model_metadata",
            model_id,
            0,
            0,
            response_size,
            0
        )
    return model.metadata


class ClusterManagerBaseStationGateway(BaseStationGateway):

    def __init__(
            self,
            cluster_manager: ClusterManager,
            profiler: Optional[Profiler],
            model_dir: str,
    ):
        self.cluster_manager = cluster_manager
        self._node_id = str(uuid.uuid4())
        self.model_dir = model_dir
        self.profiler = profiler

    def register_node(
            self,
            initial_df: Optional[pd.DataFrame],
            input_features: list[str],
            output_features: list[str]
    ) -> SensorKnowledgeInitialization:
        initialization = self.cluster_manager.register_node(input_features, output_features, initial_df)
        self._node_id = initialization.node_id
        return initialization

    def sync(self, dt: datetime.datetime) -> SensorKnowledgeUpdate:
        response = self.cluster_manager.sync(self._node_id)
        response_body = response.to_dict()
        response_body_size = _estimate_payload_size(response_body)
        self.profiler.add_log_entry(
            datetime.datetime.now(),
            0,
            "sync",
            dt.isoformat(),
            0,
            0,
            response_body_size,
            0
        )
        return response

    def send_violation(
            self,
            dt: datetime.datetime,
            configuration_id: str,
            measurement: pd.Series,
            data: pd.DataFrame
    ) -> SensorKnowledgeUpdate:
        knowledge = self.cluster_manager.knowledge
        node = knowledge.get_node(self._node_id)
        model = knowledge.model_manager.get_model(configuration_id)
        node.set_active_model(model, dt)
        serialized_data = _serialize_df(data)

        request_body = {
            'timestamp': dt.isoformat(),
            'measurement': measurement.to_dict(),
            'data': serialized_data
        }
        request_body_size = _estimate_payload_size(request_body)

        response = self.cluster_manager.handle_violation(self._node_id, dt, measurement, data, configuration_id)

        response_body = response.to_dict()
        response_body_size = _estimate_payload_size(response_body)
        self.profiler.add_log_entry(
            datetime.datetime.now(),
            0,
            "send_violation",
            dt.isoformat(),
            0,
            0,
            response_body_size,
            request_body_size
        )
        return response

    def send_update(
            self,
            dt: datetime.datetime,
            data: pd.DataFrame,
            configuration_id: str
    ) -> SensorKnowledgeUpdate:
        knowledge = self.cluster_manager.knowledge
        node = knowledge.get_node(self._node_id)
        model = knowledge.model_manager.get_model(configuration_id)
        node.set_active_model(model, dt)
        serialized_data = _serialize_df(data)

        request_body = {
            'timestamp': dt.isoformat(),
            'data': serialized_data,
        }
        request_body_size = _estimate_payload_size(request_body)

        response = self.cluster_manager.handle_horizon_update(self._node_id, dt, data, configuration_id)

        response_body = response.to_dict()
        response_body_size = _estimate_payload_size(response_body)
        self.profiler.add_log_entry(
            datetime.datetime.now(),
            0,
            "send_update",
            dt.isoformat(),
            0,
            0,
            response_body_size,
            request_body_size
        )

        return response

    def send_measurement(
            self,
            dt: datetime.datetime,
            measurement: pd.Series,
            configuration_id: str
    ) -> SensorKnowledgeUpdate:
        knowledge = self.cluster_manager.knowledge
        node = knowledge.get_node(self._node_id)
        model = knowledge.model_manager.get_model(configuration_id)
        node.set_active_model(model, dt)
        return self.cluster_manager.add_measurement(self._node_id, dt, measurement)

    def get_model_metadata(self, model_id: str) -> ModelMetadata:
        return _get_model_metadata(self.model_dir, model_id, self.profiler)

    def fetch_model(self, model_id: str) -> bytes:
        return _fetch_model(self.model_dir, model_id, self.profiler)
