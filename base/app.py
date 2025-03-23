import datetime
import logging
import os
import time

import pandas as pd
from flask import request, g

from base import app
from base.base_station_analyzer import PredictorGoalsBaseStationAnalyzer
from base.base_station_executor import SequentialBaseStationExecutor
from base.base_station_knowledge import BaseStationKnowledge
from base.base_station_manager import BaseStationManager
from base.base_station_monitor import ViolationsBaseStationMonitor
from base.base_station_planner import PortfolioBaseStationPlanner
from base.learning_strategy import RetrainStrategy
from base.model_manager import ModelManager
from base.model_trainer import ModelTrainer
from base.portfolio_adaptation_goal import DeployOnceAdaptationGoal
from common.predictor_adaptation_goal import ViolationRateAdaptationGoal
from common.resource_profiler import init_profiler, get_profiler, Profiler

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

model_trainer = ModelTrainer()
learning_strategy = RetrainStrategy(model_trainer, model_manager)
cluster_goals = [DeployOnceAdaptationGoal("deploy_once")]
node_goals = [ViolationRateAdaptationGoal("violations_goal", "TL", 1, datetime.timedelta(hours=1), 1)]
knowledge = BaseStationKnowledge(
    "base_station", model_manager, training_df, learning_strategy, cluster_goals, node_goals
)
executor = SequentialBaseStationExecutor(knowledge)
planner = PortfolioBaseStationPlanner(knowledge, executor)
analyzer = PredictorGoalsBaseStationAnalyzer(knowledge, planner)
monitor = ViolationsBaseStationMonitor(knowledge, analyzer)
cluster_manager = BaseStationManager("default_cluster", knowledge, monitor)


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
