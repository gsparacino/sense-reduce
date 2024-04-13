import logging
import logging
import os

from common import EventLogger

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
