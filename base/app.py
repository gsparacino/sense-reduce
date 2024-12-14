import datetime
import logging
import os
import time
import uuid

import pandas as pd
from flask import request, g, send_file

from base import app
from base.cluster_adaptation_goal import DeployOnceAdaptationGoal
from base.cluster_analyzer import PortfolioClusterAnalyzer
from base.cluster_executor import SequentialClusterExecutor
from base.cluster_knowledge import ClusterKnowledge
from base.cluster_manager import ClusterManager
from base.cluster_monitor import ViolationsClusterMonitor
from base.cluster_planner import PortfolioClusterPlanner
from base.dense_model_trainer import DenseModelTrainer
from base.learning_strategy import RetrainStrategy
from base.model_manager import ModelManager
from common.resource_profiler import init_profiler, get_profiler, Profiler
from common.sensor_adaptation_goal import TimeWindowAdaptationGoal

logging.basicConfig(level=logging.DEBUG)
base_dir = os.path.abspath(os.path.dirname(__file__))

# configure the parameters of the initial model and the strategies applied by the base station
model_dir = os.path.join(base_dir, app.config['MODEL_DIR'])
base_model_id = app.config['BASE_MODEL_UUID']

model_manager = ModelManager(str(model_dir), base_model_id)
base_model = model_manager.base_model

# define the dataset for training the model(s)
training_df_path = os.path.join(base_dir, app.config['TRAINING_DF'])
training_df = pd.read_pickle(str(training_df_path))
training_df = training_df[base_model.metadata.input_features]
logging.debug(f'Loaded the training dataset from "{app.config["TRAINING_DF"]}"')

init_profiler("logs")
profiler: Profiler = get_profiler()

# TODO: make MAPEK models configurable
model_trainer = DenseModelTrainer()
learning_strategy = RetrainStrategy(model_trainer, model_manager)
cluster_goals = [DeployOnceAdaptationGoal("deploy_once")]
node_goals = [TimeWindowAdaptationGoal("violations_goal", "TL", 1, datetime.timedelta(hours=1), 1)]
knowledge = ClusterKnowledge(
    "base_station", model_manager, training_df, learning_strategy, cluster_goals, node_goals
)
executor = SequentialClusterExecutor(knowledge)
planner = PortfolioClusterPlanner(knowledge, executor)
analyzer = PortfolioClusterAnalyzer(knowledge, planner)
monitor = ViolationsClusterMonitor(knowledge, analyzer)
cluster_manager = ClusterManager("default_cluster", knowledge, monitor)


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
            response_size,
            g.request_size
        )

    return response


@app.post("/nodes")
def register_node():
    """Registers a new node and returns the initial configuration for the node."""
    node_id = str(uuid.uuid4())
    body = request.get_json(force=True)
    input_features = body['input_features']
    output_features = body['output_features']
    initial_df_json = body.get('initial_df')
    initial_df = pd.read_json(initial_df_json) if initial_df_json is not None else None
    initialization = cluster_manager.register_node(input_features, output_features, initial_df)
    logging.info(f'New node registration with ID={node_id}')
    return initialization.to_dict()


@app.post("/nodes/<string:node_id>/violation")
def post_violation(node_id: str):
    body = request.get_json(force=True)
    dt = datetime.datetime.fromisoformat(body['timestamp'])
    logging.info(f'Received violation message from node {node_id} @ {dt.isoformat()}')
    # TODO: implement actual MAPEK logic on BS
    payload = dict()
    payload['portfolio'] = []  # TODO
    payload['adaptation_goals'] = []  # TODO
    return payload


@app.post("/nodes/<string:node_id>/update")
def post_update(node_id: str):
    """Called when a sensor node reaches the end of its prediction horizon."""
    body = request.get_json(force=True)
    dt = datetime.datetime.fromisoformat(body['timestamp'])
    logging.info(f'Received update message from node {node_id} @ {dt.isoformat()}')
    # TODO: implement actual MAPEK logic on BS
    payload = dict()
    payload['portfolio'] = []  # TODO
    payload['adaptation_goals'] = []  # TODO
    return payload


@app.post("/nodes/<string:node_id>/measurement")
def post_measurement(node_id: str):
    """Endpoint for a node to communicate a measurement."""
    body = request.get_json(force=True)
    dt = datetime.datetime.fromisoformat(body['timestamp'])
    logging.info(f'Node {node_id} sent a measurement @ {dt.isoformat()}')
    # TODO: implement actual MAPEK logic on BS
    payload = dict()
    payload['portfolio'] = []  # TODO
    payload['adaptation_goals'] = []  # TODO
    return payload


@app.get("/nodes/<string:node_id>/models/<string:model_id>")
def get_model(node_id: str, model_id: str):
    logging.info(f'Node {node_id} requested model {model_id}')
    model_file_path = model_manager.get_model_tflite_file_path(model_id)
    return send_file(model_file_path)


@app.get("/nodes/<string:node_id>/models/<string:model_id>/metadata")
def get_model_metadata(node_id: str, model_id: str):
    logging.info(f'Node {node_id} requested model metadata of {model_id}')
    metadata = model_manager.get_model(model_id).metadata
    return metadata.to_dict()


@app.get("/ping")
def ping():
    """Simple GET endpoint for health checks."""
    logging.info(f'GET /ping, request: {request}')
    return "pong"
