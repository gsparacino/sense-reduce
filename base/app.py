import logging
import os

import pandas as pd
from flask import request, send_file

from base import app, config
from base.cluster_manager import ClusterManager
from common import ThresholdMetric, LogEvent, LogEventType

BASEDIR = os.path.abspath(os.path.dirname(__file__))
event_logger = config.event_logger

cluster_manager = ClusterManager(config)


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
    payload['portfolio'] = cluster_manager.get_models_in_portfolio()
    logging.debug(f'Responding to new node with payload: {payload}')
    return payload


@app.post("/nodes/<string:node_id>/models/new")
def post_new_model_request(node_id: str):
    logging.info(f'Node {node_id} requested a new model')
    body = request.get_json(force=True)

    if body.get('measurements') is not None:
        measurements: pd.DataFrame = pd.read_json(body.get('measurements'))
        cluster_manager.add_measurements(node_id, measurements)

    new_model = cluster_manager.handle_new_model_request(node_id)
    payload = dict()
    if new_model is not None:
        payload['model_metadata'] = new_model.metadata.to_dict()
    logging.debug(f'Responding to node {node_id} with payload: {payload}')
    return payload


@app.post("/nodes/<string:node_id>/violation")
def post_violation(node_id: str):
    logging.info(f'Node {node_id} sent a violation notification')
    body = request.get_json(force=True)
    if body.get('measurements') is not None:
        measurements: pd.DataFrame = pd.read_json(body.get('measurements'))
        cluster_manager.add_measurements(node_id, measurements)
        cluster_manager.handle_new_model_request(node_id)
    payload = dict()
    payload['portfolio'] = cluster_manager.get_models_in_portfolio()
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
    payload = dict()
    payload['portfolio'] = cluster_manager.get_models_in_portfolio()
    return payload


@app.get("/ping")
def ping():
    """Simple GET endpoint for health checks."""
    logging.info(f'GET /ping, request: {request}')
    return "pong"
