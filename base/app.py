import logging
import os

import pandas as pd
from flask import request, send_file

from base import app, config
from base.adaptive_strategy import DefaultAdaptiveStrategy
from base.cluster_manager import ClusterManager
from base.model_manager import ModelManager
from base.model_trainer import DefaultModelTrainer
from common import ThresholdMetric, LogEvent, LogEventType

BASEDIR = os.path.abspath(os.path.dirname(__file__))
event_logger = config.event_logger

model_manager = ModelManager(config)
model_trainer = DefaultModelTrainer(epochs=2)
adaptive_strategy = DefaultAdaptiveStrategy(config, model_manager, model_trainer)
cluster_manager = ClusterManager(config, model_manager, model_trainer, adaptive_strategy)


@app.post("/nodes")
def register_node():
    """Registers a new node and returns the model metadata and initial data for the node."""
    body: dict = request.get_json(force=True)
    node_id = body["node_id"]
    threshold_metric = ThresholdMetric.from_dict(body['threshold_metric'])

    event_logger.log_event(LogEvent(node_id, LogEventType.REGISTRATION))

    cluster_manager.add_node(node_id, threshold_metric)
    model = cluster_manager.get_current_model(node_id)
    initial_df = cluster_manager.get_measurements(node_id)

    payload = dict()
    payload['model_metadata'] = model.metadata.to_dict()
    payload['initial_df'] = initial_df.to_json()
    payload['portfolio'] = cluster_manager.get_recommended_models(node_id)
    logging.debug(f'Responding to new node with payload: {payload}')
    return payload


@app.post("/nodes/<string:node_id>/violation")
def post_violation(node_id: str):
    logging.info(f'Node {node_id} sent a violation notification')
    body = request.get_json(force=True)
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
    model_file_path = cluster_manager.get_model_upload_path(model_id)
    event_logger.log_event(LogEvent(node_id, LogEventType.MODEL_UPDATE, f"{model_id}"))
    return send_file(model_file_path)


@app.get("/nodes/<string:node_id>/models/<string:model_id>/metadata")
def get_model_metadata(node_id: str, model_id: str):
    logging.info(f'Node {node_id} requested model {model_id}')
    metadata = cluster_manager.get_model_metadata(model_id)
    return metadata.to_dict()


@app.post("/nodes/<string:node_id>/sync")
def sync(node_id: str):
    logging.info(f'Node {node_id} sent synchronization request')
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
