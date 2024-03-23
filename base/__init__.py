from flask import Flask

# adapt to you needs
app = Flask('base', static_folder='models')
app.config['MODEL_DIR'] = 'E:\\Docs\\src\\sense-reduce\\models'
# app.config['MODEL_DIR'] = '/usr/local/app/models'
app.config['LOG_DIR'] = 'E:\\Docs\\src\\sense-reduce\\log'
# app.config['LOG_DIR'] = '/usr/local/app/log'
app.config['BASE_MODEL_UUID'] = 'base-model'
app.config['TRAINING_DF'] = 'data/training.pickle'
