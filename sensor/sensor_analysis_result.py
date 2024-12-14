import pandas as pd

from sensor.predictor_configuration import PredictorConfiguration


class PredictorAnalysisResult:

    def __init__(self, configuration: PredictorConfiguration, evaluation: pd.Series):
        self.configuration = configuration
        self.evaluation = evaluation
