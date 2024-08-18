import numpy as np
import pandas as pd

from common import Predictor, ThresholdMetric

PREDICTION_COLUMN = 'prediction'
ERROR_COLUMN = 'error'
MODEL_COLUMN = 'model'


class ModelSimulator:

    def __init__(self, predictor: Predictor, threshold_metric: ThresholdMetric, test_data: pd.DataFrame):
        self.predictor = predictor
        self.test_data = test_data
        self.threshold_metric = threshold_metric
        self.simulation_results = self._init_simulation_results(test_data)

    @staticmethod
    def _init_simulation_results(test_data: pd.DataFrame) -> pd.DataFrame:
        columns = np.concatenate((test_data.columns.values, [PREDICTION_COLUMN, ERROR_COLUMN, MODEL_COLUMN]))
        simulation_results = pd.DataFrame(columns=columns)
        simulation_results.index = pd.to_datetime(simulation_results.index)
        return simulation_results

    def run(self) -> pd.DataFrame:
        for index, data in self.test_data.iterrows():
            data: np.ndarray
            index: pd.to_datetime
            if not self.predictor.in_prediction_horizon(index):
                self.predictor.update_prediction_horizon(index)
            prediction = self.predictor.get_prediction_at(index)
            score = self.threshold_metric.threshold_score(data, prediction)
            result = data.copy()
            result[PREDICTION_COLUMN] = prediction['TL']
            result[ERROR_COLUMN] = score
            result[MODEL_COLUMN] = self.predictor.model_metadata.uuid
            self.simulation_results.loc[index] = result
            if self.threshold_metric.is_threshold_violation(data, prediction):
                self.predictor.update_prediction_horizon(index)
                prediction = self.predictor.get_prediction_at(index).to_numpy()
                self.predictor.adjust_to_measurement(index, data, prediction)

        return self.simulation_results
