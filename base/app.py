import json
import logging
import os
from datetime import datetime

import pandas as pd
import tensorflow as tf
from flask import request, send_file

from base import app, config
from base.deployment_strategy import CorrectiveStrategy
from base.learning_strategy import RetrainStrategy
from base.model import Model
from base.node_manager import NodeManager
from common import ThresholdMetric, LogEvent, LogEventType, ModelMetadata

BASEDIR = os.path.abspath(os.path.dirname(__file__))
event_logger = config.event_logger

# define the dataset for training the model(s)
path = os.path.join(BASEDIR, config.data_dir, config.initial_data_pickle)
logging.info(f'Loading the training dataset from "{path}"')
training_df = pd.read_pickle(path)

# configure the parameters of the initial model and the strategies applied by the base station
path = os.path.join(BASEDIR, config.model_dir, config.base_model_id)
logging.info(f'Loading the base model from "{path}"')
with open(os.path.join(path, 'metadata.json'), 'r') as fp:
    metadata = ModelMetadata.from_dict(json.load(fp))
    base_model = Model(tf.keras.models.load_model(path), metadata)

# TODO: make the strategies configurable
cl_strategy = RetrainStrategy(epochs=10, patience=1)
# cl_strategy = NoUpdateStrategy()
cl_strategy.add_model(base_model)
deploy_strategy = CorrectiveStrategy()
node_manager = NodeManager(cl_strategy, deploy_strategy, os.path.join(BASEDIR, config.model_dir).__str__())


@app.post("/register/<string:node_id>")
def register_node(node_id: str):
    """Registers a new node and returns the model metadata and initial data for the node."""
    body: dict = request.get_json(force=True)
    threshold_metric = ThresholdMetric.from_dict(body['threshold_metric'])
    prediction_period_s = body['prediction_period_s']
    if body.get('initial_df') is None:
        initial_df = training_df
    else:
        initial_df = pd.read_json(body.get('initial_df'))
    logging.info(f'New node with ID={node_id}, threshold={threshold_metric}, and initial_df={initial_df} registered')

    # TODO: test
    node = node_manager.add_node(node_id, threshold_metric, initial_df, datetime.now(), prediction_period_s)
    payload = dict()
    payload['model_metadata'] = node.predictor.model_metadata.to_dict()
    payload['initial_df'] = initial_df.to_json()
    logging.debug(f'Responding to new node with payload: {payload}')
    event_logger.log_event(LogEvent(node_id, LogEventType.REGISTRATION))
    return payload


@app.get("/models/<string:node_id>")
def get_default_model(node_id: str):
    logging.info(f'Node "{node_id}" requested its model')
    # TODO: node_id must be UUID and should exist
    model_file = node_manager.on_model_deployment(node_id, datetime.now())
    event_logger.log_event(LogEvent(node_id, LogEventType.MODEL_UPDATE, "Node request"))
    return send_file(model_file)


@app.get("/models/<string:node_id>/<string:model_id>")
def get_model(node_id: str, model_id: str):
    logging.info(f'Node "{node_id}" requested model {model_id}')
    # TODO: use model_id to pick a model from the node's porfolio
    model_file = node_manager.on_model_deployment(node_id, datetime.now())
    event_logger.log_event(LogEvent(node_id, LogEventType.MODEL_UPDATE, "Node request"))
    return send_file(model_file)


@app.post("/violation/<string:node_id>")
def post_violation(node_id: str):
    body = request.get_json(force=True)
    logging.info(f'Received violation message from node {node_id}: {body}')
    dt = datetime.fromisoformat(body['timestamp'])
    event_logger.log_event(LogEvent(node_id, LogEventType.VIOLATION))
    event_logger.log_event(LogEvent(node_id, LogEventType.MEASUREMENT,
                                    "All measurements since the beginning of the latest prediction horizon")
                           )
    new_model = node_manager.on_threshold_violation(node_id, dt, body['measurement'], pd.read_json(body['data']))
    return {'model_metadata': None if new_model is None else new_model.to_dict()}, 201


@app.post("/update/<string:node_id>")
def post_update(node_id: str):
    """Called when a sensor node reaches the end of its prediction horizon."""
    body = request.get_json(force=True)
    logging.info(f'Received update message from node {node_id} with body {body}')
    dt = datetime.fromisoformat(body['timestamp'])
    event_logger.log_event(
        LogEvent(node_id, LogEventType.MEASUREMENT, "All measurements within the latest prediction horizon")
    )
    event_logger.log_event(LogEvent(node_id, LogEventType.HORIZON_UPDATE))
    new_model = node_manager.on_horizon_update(node_id, dt, pd.read_json(body['data']))
    if new_model is not None:
        event_logger.log_event(LogEvent(node_id, LogEventType.MODEL_UPDATE, "Horizon update"))
    return {'model_metadata': None if new_model is None else new_model.to_dict()}, 201


@app.post("/measurement/<string:node_id>")
def post_measurement(node_id: str):
    """Endpoint for a node to communicate a measurement."""
    body = request.get_json(force=True)
    logging.info(f'Node {node_id} sent a measurement: {body}')

    dt = datetime.fromisoformat(body['timestamp'])
    node_manager.get_node(node_id).add_measurement(dt, body['measurement'])
    event_logger.log_event(LogEvent(node_id, LogEventType.MEASUREMENT, "Single measurement"))
    return '', 201


@app.get("/prediction/<string:node_id>")
def get_prediction(node_id: str):
    """Returns the current prediction for the specified node."""
    logging.info(f'Request for temperature prediction at node {node_id}')
    event_logger.log_event(LogEvent(node_id, LogEventType.PREDICTION))
    return node_manager.get_prediction_at(node_id, datetime.now()).to_json()


@app.get("/nodes")
def get_nodes():
    """Returns all currently registered nodes of the node manager."""
    logging.info(f'Request for registered nodes')

    return {
        'node_ids': node_manager.node_ids
    }


@app.get("/ping")
def ping():
    """Simple GET endpoint for health checks."""
    logging.info(f'GET /ping, request: {request}')
    return "pong"
