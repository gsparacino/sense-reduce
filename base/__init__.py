from flask import Flask

# adapt to you needs
app = Flask('base', static_folder='models')
app.config['MODEL_DIR'] = 'models'
app.config['BASE_MODEL_UUID'] = 'base-model'
app.config['TRAINING_DF'] = 'data/training.pickle'
