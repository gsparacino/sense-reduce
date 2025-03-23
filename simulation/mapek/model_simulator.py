import datetime
from datetime import timedelta

import numpy as np
import pandas as pd

from common.predictor import Predictor
from common.threshold_metric import ThresholdMetric

PREDICTION_COLUMN = 'prediction'
ERROR_COLUMN = 'error'
MODEL_COLUMN = 'model'
VIOLATION_COLUMN = 'violation'


class ModelSimulator:

    def __init__(self,
                 predictor: Predictor,
                 threshold_metric: ThresholdMetric,
                 test_data: pd.DataFrame,
                 resolution_in_seconds: int
                 ):
        self.predictor = predictor
        self.test_data = test_data
        self.threshold_metric = threshold_metric
        self.simulation_results = self._init_simulation_results(test_data)
        self.resolution_in_seconds = resolution_in_seconds

    @staticmethod
    def _init_simulation_results(test_data: pd.DataFrame) -> pd.DataFrame:
        columns = np.concatenate(
            (test_data.columns.values, [PREDICTION_COLUMN, ERROR_COLUMN, MODEL_COLUMN, VIOLATION_COLUMN])
        )
        simulation_results = pd.DataFrame(columns=columns)
        simulation_results.index = pd.to_datetime(simulation_results.index)
        return simulation_results

    def run(self) -> pd.DataFrame:
        iterate_df: pd.DataFrame = (
            self.test_data.resample(timedelta(seconds=self.resolution_in_seconds)).interpolate(method='time')
        )

        for index, data in iterate_df.iterrows():
            data: np.ndarray
            index: datetime.datetime

            self.predictor.add_measurement(index, data)

            if not self.predictor.in_prediction_horizon(index):
                self.predictor.update_prediction_horizon(index)

            prediction = self.predictor.get_prediction_at(index)

            prediction_array = prediction.to_numpy()
            measurement_array = np.asarray(data)
            score = self.threshold_metric.threshold_score(data, prediction_array)
            result = data.copy()
            result[PREDICTION_COLUMN] = prediction['TL']
            result[ERROR_COLUMN] = score
            result[MODEL_COLUMN] = self.predictor.model_metadata.uuid
            if self.threshold_metric.is_threshold_violation(measurement_array, prediction_array):
                result[VIOLATION_COLUMN] = 1
                self.predictor.update_prediction_horizon(index)
                prediction = self.predictor.get_prediction_at(index).to_numpy()
                self.predictor.adjust_to_measurement(index, data, prediction)
            else:
                result[VIOLATION_COLUMN] = 0
            self.simulation_results.loc[index] = result
        return self.simulation_results
