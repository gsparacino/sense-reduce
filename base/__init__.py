import os

import yaml

from base.config import Config

# Load configuration yaml
basedir = os.path.abspath(os.path.dirname(__file__))
config_path = os.environ.get('CONFIG_PATH') or os.path.join(basedir, 'config', 'config.yaml')

with open(config_path) as config_file:
    config_data = yaml.safe_load(config_file)
    config: Config = Config(**config_data)
