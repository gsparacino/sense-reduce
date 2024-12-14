import datetime
import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from common.data_reduction_strategy import DataReductionStrategy
from common.predictor import Predictor


class SensorAdaptationGoal(ABC):

    def __init__(self, goal_id: str):
        self.goal_id = goal_id

    @abstractmethod
    def is_violation(self, measurement: pd.Series, prediction: pd.Series) -> bool:
        pass

    @abstractmethod
    def requires_adaptation(self, predictor: Predictor, dt: datetime.datetime) -> bool:
        pass

    @abstractmethod
    def evaluate(
            self,
            predictor: Predictor,
            data_reduction_strategy: DataReductionStrategy,
            dt: datetime.datetime
    ) -> float:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @classmethod
    def from_dict(cls, data: dict) -> 'SensorAdaptationGoal':
        goal_type = data['type']

        match goal_type:
            case TimeWindowAdaptationGoal.type_id:
                return TimeWindowAdaptationGoal.deserialize(data)
            case _:
                raise ValueError(f'PredictorAdaptationGoal type {goal_type} not supported')


class TimeWindowAdaptationGoal(SensorAdaptationGoal):
    type_id: str = 'time_window'

    def __init__(
            self,
            goal_id: str,
            metric_id: str,
            threshold: float,
            time_window: datetime.timedelta,
            max_violations: int,
    ):
        super().__init__(goal_id)
        self.metric_id = metric_id
        self.threshold = threshold
        self.time_window = time_window
        self.max_violations = max_violations

    def to_dict(self) -> dict:
        return {
            'type': TimeWindowAdaptationGoal.type_id,
            'goal_id': self.goal_id,
            'metric_id': self.metric_id,
            'threshold': self.threshold,
            'time_window_s': self.time_window.seconds,
            'max_violations': self.max_violations,
        }

    @classmethod
    def deserialize(cls, data: dict) -> 'TimeWindowAdaptationGoal':
        if data['type'] != TimeWindowAdaptationGoal.type_id:
            raise ValueError(f"Invalid type id {data['type']} for {cls.__name__}")

        return TimeWindowAdaptationGoal(
            data.get('goal_id'),
            data.get('metric_id'),
            data.get('threshold'),
            data.get('time_window_s'),
            data.get('max_violations'),
        )

    def is_violation(self, measurement: pd.Series, prediction: pd.Series) -> bool:
        return self._diff(measurement, prediction) > self.threshold

    def requires_adaptation(self, predictor: Predictor, dt: datetime.datetime) -> bool:
        return self._count_violations(predictor, dt) >= self.max_violations

    def evaluate(
            self,
            predictor: Predictor,
            data_reduction_strategy: DataReductionStrategy,
            dt: datetime.datetime
    ) -> float:
        measurements = self._get_test_measurements(dt, predictor)
        if len(measurements) == 0:
            return 0
        max_diff = len(measurements) * self.threshold
        diff = 0
        since = (dt - self.time_window)
        predictor.update_prediction_horizon(since)
        for timestamp in measurements.index:
            measurement = measurements.loc[timestamp]
            prediction = predictor.get_prediction_at(timestamp)
            diff += self._diff(measurement, prediction)
            if diff > max_diff:
                return 0
        return 1 - (diff / max_diff)

    def _count_violations(self, predictor: Predictor, until: datetime.datetime) -> int:
        threshold_dt = (until - self.time_window)
        violations = predictor.data.get_violations_of_model(predictor.model_metadata.uuid)
        violations = violations[(violations.index >= threshold_dt) & (violations.index <= until)]
        return len(violations)

    def _get_test_measurements(self, dt: datetime.datetime, predictor: Predictor):
        threshold_dt = (dt - self.time_window)
        measurements = predictor.data.get_measurements()
        measurements = measurements[(measurements.index >= threshold_dt) & (measurements.index <= dt)]
        return measurements

    def _diff(self, measurement: pd.Series, prediction: pd.Series) -> float:
        if self.metric_id not in prediction.index:
            logging.warning(
                f'({self.goal_id}): {self.metric_id} not found in prediction'
            )
            return 0
        if self.metric_id not in measurement.index:
            logging.warning(
                f'({self.goal_id}): {self.metric_id} not found in measurement'
            )
            return 0
        return np.linalg.norm(measurement[self.metric_id] - prediction[self.metric_id])
