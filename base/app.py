import datetime
import json
import logging
import os
import time

import pandas as pd
from flask import request, send_file, g, Flask

from base import config
from base.base_adaptive_strategy import DefaultBaseStationAdaptiveStrategy
from base.cluster_manager import ClusterManager
from base.model_manager import ModelManager
from base.model_trainer import RetrainLearningStrategy
from common import ThresholdMetric, LogEvent, LogEventType

BASEDIR = os.path.abspath(os.path.dirname(__file__))

model_manager = ModelManager(config)
# TODO: make number of epochs configurable
model_trainer = RetrainLearningStrategy(epochs=10)
adaptive_strategy = DefaultBaseStationAdaptiveStrategy(config, model_manager, model_trainer)
cluster_manager = ClusterManager(config, model_manager, model_trainer, adaptive_strategy)

profiler = config.profiler
event_logger = config.event_logger

# Create a Flask application
app = Flask('base', static_folder=config.model_dir)


@app.before_request
def before_request():
    if profiler is not None:
        if request.data:
            request_size = len(request.data)
            g.request_size = request_size
        else:
            g.request_size = 0
        g.timestamp = datetime.datetime.now()
        g.start_time = time.time()


@app.after_request
def after_request(response):
    if profiler is not None:
        response_size = int(response.headers.get('Content-Length'))

        profiler.add_log_entry(
            g.timestamp,
            time.time() - g.start_time,
            request.endpoint,
            f'{request.method} {request.path} ',
            0,
            0,
            g.request_size,
            response_size
        )

    return response


@app.post("/nodes/<string:node_id>/measurement")
def add_measurement(node_id: str):
    logging.info(f'Node {node_id} sent new measurement: {request.data}')
    event_logger.log_event(LogEvent(node_id, LogEventType.MEASUREMENT))
    body = request.get_json(force=True)
    data = json.loads(body.get('measurement'))
    timestamp = datetime.datetime.strptime(body.get('timestamp'), '%Y-%m-%dT%H:%M:%S.%f')
    new_measurement = pd.DataFrame(data=[data], index=[pd.to_datetime(timestamp)])
    cluster_manager.add_measurements(node_id, new_measurement)
    return '', 200


@app.post("/nodes")
def register_node():
    """Registers a new node and returns the model metadata and initial data for the node."""
    body: dict = request.get_json(force=True)
    node_id = body["node_id"]
    event_logger.log_event(LogEvent(node_id, LogEventType.REGISTRATION))
    threshold_metric = ThresholdMetric.from_dict(body['threshold_metric'])

    logging.info(f'Node registration request: {node_id}')

    cluster_manager.add_node(node_id, threshold_metric)
    model = cluster_manager.get_current_model(node_id)
    initial_df = cluster_manager.get_measurements(node_id)

    payload = dict()
    payload['model_metadata'] = model.metadata.to_dict()
    payload['portfolio'] = cluster_manager.get_recommended_models(node_id)
    logging.debug(f'Responding to new node with payload: {payload}')
    return payload


@app.post("/nodes/<string:node_id>/violation")
def post_violation(node_id: str):
    logging.info(f'Node {node_id} sent a violation notification')
    body = request.get_json(force=True)
    event_logger.log_event(LogEvent(node_id, LogEventType.VIOLATION))
    cluster_manager.add_violation(node_id, body['timestamp'], body['model'])
    if body.get('measurements') is not None:
        measurements: pd.DataFrame = pd.read_json(body.get('measurements'))
        cluster_manager.add_measurements(node_id, measurements)
    new_model_flag: bool = body['needs_new_model']
    if new_model_flag:
        logging.info(f'Node {node_id} requested a new model')
        node_portfolio = body['portfolio']
        cluster_manager.handle_new_model_request(node_id, node_portfolio)

    payload = dict()
    payload['portfolio'] = cluster_manager.get_recommended_models(node_id)
    return payload


@app.get("/nodes/<string:node_id>/models/<string:model_id>")
def get_model(node_id: str, model_id: str):
    logging.info(f'Node {node_id} requested model {model_id}')
    event_logger.log_event(LogEvent(node_id, LogEventType.MODEL_DEPLOYMENT, model_id))
    model_file_path = cluster_manager.get_model_upload_path(model_id)
    return send_file(model_file_path)


@app.get("/nodes/<string:node_id>/models/<string:model_id>/metadata")
def get_model_metadata(node_id: str, model_id: str):
    logging.info(f'Node {node_id} requested model metadata of {model_id}')
    metadata = cluster_manager.get_model_metadata(model_id)
    return metadata.to_dict()


@app.post("/nodes/<string:node_id>/sync")
def sync(node_id: str):
    logging.info(f'Node {node_id} sent synchronization request')
    event_logger.log_event(LogEvent(node_id, LogEventType.SYNC))
    body = request.get_json(force=True)
    if body.get('measurements') is not None:
        measurements: pd.DataFrame = pd.read_json(body.get('measurements'))
        cluster_manager.add_measurements(node_id, measurements)
    cluster_manager.set_current_model(node_id, body['model'])
    payload = dict()
    payload['portfolio'] = cluster_manager.get_recommended_models(node_id)
    return payload


@app.get("/ping")
def ping():
    """Simple GET endpoint for health checks."""
    logging.info(f'GET /ping, request: {request}')
    return "pong"
