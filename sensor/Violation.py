import datetime

import numpy as np

from common import Predictor


class Violation:

    def __init__(self, node_id: str,
                 timestamp: datetime.datetime,
                 predictor: Predictor,
                 measurement: np.array,
                 prediction: np.array):
        self.node_id = node_id
        self.timestamp = timestamp
        self.predictor = predictor
        self.measurement = measurement
        self.prediction = prediction
