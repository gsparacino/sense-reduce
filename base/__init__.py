from flask import Flask

# adapt to you needs
app = Flask('base', static_folder='models')
app.config['MODEL_DIR'] = 'models'
app.config['BASE_MODEL_UUID'] = 'zamg_vienna_2010_2019_simple_dense'
app.config['TRAINING_DF'] = 'data/zamg_vienna_hourly.pickle'
