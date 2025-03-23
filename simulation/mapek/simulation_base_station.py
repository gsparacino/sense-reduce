import datetime
import json
import logging
import os
import sys
import threading
import time
import uuid
from typing import Optional

import pandas as pd
import requests
from flask import Flask, g, request
from keras.src.utils.io_utils import disable_interactive_logging
from requests import Response
from werkzeug.serving import make_server

from base.base_station_manager import BaseStationManager
from base.model_manager import ModelManager
from common.model_metadata import ModelMetadata
from common.model_utils import load_model_from_savemodel, to_tflite_model_bytes
from common.resource_profiler import Profiler, init_profiler
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
            cluster_manager: BaseStationManager,
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

    def synchronize(self, dt: datetime.datetime) -> SensorKnowledgeUpdate:
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

    def send_horizon_update(
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


class ServerThread(threading.Thread):
    def __init__(self, host: str, port: int, app: Flask):
        threading.Thread.__init__(self)
        self._server = make_server(host=host, port=port, app=app)

    def run(self):
        logging.info("Starting server thread...")
        disable_interactive_logging()
        self._server.serve_forever()

    def stop(self):
        logging.info("Gracefully stopping server thread...")
        self._server.shutdown()

    def kill(self):
        logging.info("Forceful shutdown of server thread...")
        try:
            sys.exit(0)
        except SystemExit:
            logging.info("Stopping...")


class SimulationBaseStation:

    def __init__(self, log_path: str, base_station_manager: BaseStationManager, model_manager: ModelManager):
        self.log_path = log_path
        self.base_station_manager = base_station_manager
        self.model_manager = model_manager
        self._server: Optional[ServerThread] = None
        self.host = '127.0.0.1'
        self.port = 5000
        self.profiler: Optional[Profiler] = None

    @property
    def address(self) -> str:
        return f"http://{self.host}:{self.port}"

    @staticmethod
    def _create_flask_app(
            profiler: Profiler,
            base_station_manager: BaseStationManager,
            model_manager: ModelManager
    ) -> Flask:
        app: Flask = Flask(__name__)

        from base.base_station_blueprint import base_station_bp
        app.register_blueprint(base_station_bp)
        app.config['BASE_STATION'] = base_station_manager
        app.config['MODEL_MANAGER'] = model_manager

        @app.before_request
        def before_request():
            if request.data:
                request_size = len(request.data)
                g.request_size = request_size
            else:
                g.request_size = 0
            g.timestamp = datetime.datetime.now()
            g.start_time = time.time()

        @app.after_request
        def after_request(response):
            response_size = int(response.headers.get('Content-Length'))

            profiler.add_log_entry(
                g.timestamp,
                time.time() - g.start_time,
                request.endpoint,
                f'{request.method} {request.path} ',
                0,
                0,
                response_size,
                g.request_size
            )

            return response

        return app

    def start(self):
        self.profiler = init_profiler(self.log_path)
        app: Flask = self._create_flask_app(
            profiler=self.profiler,
            base_station_manager=self.base_station_manager,
            model_manager=self.model_manager
        )
        self._server = ServerThread(self.host, self.port, app)
        self._server.start()

        disable_interactive_logging()

        time.sleep(2)

        while not self.is_ready():
            time.sleep(2)

    def stop(self):
        if self._server is None:
            pass

        self._server.stop()

        counter = 0

        while self.is_ready() and counter < 10:
            time.sleep(2)
            counter += 1

        self._server.kill()

    def is_ready(self) -> bool:
        try:
            response: Response = requests.get(f"http://{self.host}:{self.port}/ping")
        except requests.exceptions.RequestException:
            return False
        return response.status_code == 200
