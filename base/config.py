import json
import logging
import os

import pandas as pd
import tensorflow as tf

from base.deployment_strategy import CorrectiveStrategy
from base.learning_strategy import RetrainStrategy
from base.model import Model
from base.node_manager import NodeManager
from common import ModelMetadata, EventLogger

BASEDIR = os.path.abspath(os.path.dirname(__file__))


class Config(object):

    def __init__(self,
                 model_dir: str, log_dir: str, data_dir: str, base_model_id: str, initial_data_pickle: str,
                 log_level: str):
        logging.basicConfig(level=logging.getLevelName(log_level))
        logging.info(f'Initializing configuration: '
                     f'model_dir = {model_dir}, '
                     f'log_dir = {log_dir}, '
                     f'data_dir = {data_dir}, '
                     f'base_model_id = {base_model_id}, '
                     f'initial_data_pickle = {initial_data_pickle}, '
                     f'log_level = {log_level}'
                     )
        self.event_logger = EventLogger(os.path.join(BASEDIR, log_dir))
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.base_model_id = base_model_id
        self.initial_data_pickle = initial_data_pickle

    @classmethod
    def load_training_data(cls) -> pd.DataFrame:
        path = os.path.join(cls._BASEDIR, 'data', 'zamg_vienna_hourly.pickle')
        logging.info(f'Loading the training dataset from "{path}"')
        return pd.read_pickle(path)

    @classmethod
    def load_base_model(cls) -> Model:
        base_model_id = 'zamg_vienna_2019_2019_simple_dense'
        logging.info(f'Loading initial model with ID={base_model_id}')
        model_path = os.path.join(cls._BASEDIR, cls.MODEL_DIR, base_model_id)
        with open(os.path.join(model_path, 'metadata.json'), 'r') as fp:
            metadata = ModelMetadata.from_dict(json.load(fp))
            return Model(tf.keras.models.load_model(model_path), metadata)

    @classmethod
    def load_node_manager(cls) -> NodeManager:
        logging.info(f'Starting Node Manager')
        cl_strategy = RetrainStrategy(epochs=10, patience=1)
        # cl_strategy = NoUpdateStrategy()
        cl_strategy.add_model(cls.BASE_MODEL)
        deploy_strategy = CorrectiveStrategy()
        return NodeManager(cl_strategy, deploy_strategy, Config.MODEL_DIR)
