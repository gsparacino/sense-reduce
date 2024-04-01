import os

import yaml
from flask import Flask

from base.config import Config

# Load configuration yaml
basedir = os.path.abspath(os.path.dirname(__file__))
file_path = os.environ.get('CONFIG_PATH') if os.environ.get('CONFIG_PATH') else os.path.join('config', 'config.yaml')
config_path = os.path.join(basedir, file_path)

with open(config_path) as config_file:
    config_data = yaml.safe_load(config_file)
    config: Config = Config(**config_data)

# Start Flask application
app = Flask('base', static_folder=config.model_dir)
