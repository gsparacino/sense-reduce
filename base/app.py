import json
import logging
import os
from datetime import datetime

import pandas as pd
import tensorflow as tf
from flask import request, send_file

from base import app
from base.deployment_strategy import DeployOnceStrategy
from base.learning_strategy import NoUpdateStrategy
from base.model import Model
from base.node_manager import NodeManager
from common import ThresholdMetric
from common.model_metadata import ModelMetadata

logging.basicConfig(level=logging.DEBUG)
base_dir = os.path.abspath(os.path.dirname(__file__))

# configure the parameters of the initial model and the strategies applied by the base station
model_dir = os.path.join(base_dir, app.config['MODEL_DIR'])
base_model_id = app.config['BASE_MODEL_UUID']
with open(os.path.join(str(model_dir), base_model_id, ModelMetadata.FILE_NAME), 'r') as fp:
    metadata = ModelMetadata.from_dict(json.load(fp))
base_model = Model(tf.keras.models.load_model(os.path.join(base_dir, app.config['MODEL_DIR'], base_model_id)), metadata)

# define the dataset for training the model(s)
training_df_path = os.path.join(base_dir, app.config['TRAINING_DF'])
training_df = pd.read_pickle(str(training_df_path))
training_df = training_df[metadata.input_features]
logging.debug(f'Loaded the training dataset from "{app.config["TRAINING_DF"]}"')

# TODO: make the strategies configurable
cl_strategy = NoUpdateStrategy()
cl_strategy.add_model(base_model)
node_manager = NodeManager(cl_strategy, DeployOnceStrategy(), str(model_dir))
logging.info(f'Loaded initial model with ID={base_model_id} and started node manager')


@app.post("/register/<string:node_id>")
def register_node(node_id: str):
    """Registers a new node and returns the model metadata and initial data for the node."""
    body: dict = request.get_json(force=True)
    threshold_metric = ThresholdMetric.from_dict(body['threshold_metric'])
    initial_df = body.get('initial_df')
    if initial_df is None:
        initial_df = training_df
    else:
        initial_df = pd.read_json(initial_df)
    logging.info(f'New node registration with ID={node_id}, threshold={threshold_metric}, and initial_df\n{initial_df}')
    node = node_manager.add_node(node_id, threshold_metric, initial_df, datetime.now())
    payload = dict()
    payload['model_metadata'] = node.predictor.model_metadata.to_dict()
    payload['initial_df'] = initial_df.to_json()
    logging.debug(f'Responding to new node with payload: {payload}')
    return payload


@app.get("/models/<string:node_id>")
def get_model(node_id: str):
    logging.info(f'Node "{node_id}" requested its model')
    # TODO: node_id must be UUID and should exist
    return send_file(node_manager.on_model_deployment(node_id, datetime.now()))


@app.post("/violation/<string:node_id>")
def post_violation(node_id: str):
    body = request.get_json(force=True)
    logging.info(f'Received violation message from node {node_id}: {body}')
    dt = datetime.fromisoformat(body['timestamp'])

    new_model = node_manager.on_threshold_violation(node_id, dt, body['measurement'], pd.read_json(body['data']))
    return {'model_metadata': None if new_model is None else new_model.to_dict()}, 201


@app.post("/update/<string:node_id>")
def post_update(node_id: str):
    """Called when a sensor node reaches the end of its prediction horizon."""
    body = request.get_json(force=True)
    logging.info(f'Received update message from node {node_id} with timestamp {body["timestamp"]}')
    dt = datetime.fromisoformat(body['timestamp'])

    new_model = node_manager.on_horizon_update(node_id, dt, pd.read_json(body['data']))
    return {'model_metadata': None if new_model is None else new_model.to_dict()}, 201


@app.post("/measurement/<string:node_id>")
def post_measurement(node_id: str):
    """Endpoint for a node to communicate a measurement."""
    body = request.get_json(force=True)
    logging.info(f'Node {node_id} sent a measurement: {body}')

    dt = datetime.fromisoformat(body['timestamp'])
    node_manager.get_node(node_id).add_measurement(dt, body['measurement'])
    return '', 201


@app.get("/prediction/<string:node_id>")
def get_prediction(node_id: str):
    """Returns the current prediction for the specified node."""
    logging.info(f'Request for temperature prediction at node {node_id}')

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
