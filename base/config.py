import logging
import os

from common import EventLogger
from common.resource_profiler import init_profiler, get_profiler

BASEDIR = os.path.abspath(os.path.dirname(__file__))


class Config(object):

    def __init__(self,
                 model_dir: str, log_dir: str, data_dir: str, base_model_id: str,
                 training_data_pickle: str, log_level: str):
        logging.basicConfig(level=logging.getLevelName(log_level))
        logging.info(f'Initializing configuration: '
                     f'model_dir = {model_dir}, '
                     f'log_dir = {log_dir}, '
                     f'data_dir = {data_dir}, '
                     f'base_model_id = {base_model_id}, '
                     f'training_data_pickle = {training_data_pickle}, '
                     f'log_level = {log_level}'
                     )
        self.event_logger = EventLogger(log_dir)
        self.model_dir = os.path.join(BASEDIR, model_dir)
        self.data_dir = os.path.join(BASEDIR, data_dir)
        self.base_model_id = base_model_id
        self.training_data_pickle_path = os.path.join(BASEDIR, data_dir, training_data_pickle)
        init_profiler(log_dir)
        self.profiler = get_profiler()
