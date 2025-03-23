import datetime
import logging
from abc import ABC, abstractmethod

import pandas as pd

from sensor.abstract_sensor import AbstractSensor
from sensor.sensor_analyzer import SensorAnalyzer
from sensor.sensor_knowledge import SensorKnowledge


class SensorMonitor(ABC):

    def __init__(self,
                 knowledge: SensorKnowledge,
                 analyzer: SensorAnalyzer,
                 ):
        self.knowledge = knowledge
        self.analyzer = analyzer

    @abstractmethod
    def monitor(self, sensor: AbstractSensor) -> None:
        pass


class MultivariateSensorMonitor(SensorMonitor):

    def __init__(self, knowledge: SensorKnowledge, analyzer: SensorAnalyzer):
        super().__init__(knowledge, analyzer)

    def monitor(self, sensor: AbstractSensor) -> None:
        # Read measurement
        dt, measurement = self.read_measurement(sensor)

        # Get prediction
        prediction = self.get_prediction(dt)

        # Check adaptation goals for violations
        knowledge = self.knowledge
        goals = knowledge.adaptation_goals
        predictor = knowledge.predictor
        if any(g.is_violation(measurement, prediction) for g in goals):
            logging.info(f"{dt.isoformat()} SN Monitor: violation.")
            predictor.data.add_violation(dt, predictor.model_metadata.uuid)
            self.analyzer.analyze(dt)

    def read_measurement(self, sensor: AbstractSensor) -> (datetime.datetime, pd.Series):
        """
        Reads a new measurement from the sensor.

        Args:
            sensor: the sensor to read the measurement from.

        Returns:
            The measurement values as a Pandas Series, and the timestamp of the measurement as a datetime.
        """
        predictor = self.knowledge.predictor
        dt, measurement = sensor.measurement
        predictor.data.add_measurement(dt, measurement.to_numpy())
        return dt, measurement

    def get_prediction(self, dt: datetime.datetime) -> pd.Series:
        """
        Returns the predicted values for the provided timestamp.

        Args:
            dt: the timestamp of the prediction to calculate.

        Returns:
            The predicted values as a Pandas Series.
        """
        knowledge = self.knowledge
        predictor = knowledge.predictor

        if not knowledge.predictor.in_prediction_horizon(dt):
            logging.info(f"{dt.isoformat()} SN Monitor: refresh prediction horizon.")
            update = knowledge.base_station.notify_horizon_update(dt, predictor)
            if update is not None:
                knowledge.update(dt, update)
            self._refresh_prediction_horizon(dt, knowledge)
        prediction = predictor.get_prediction_at(dt)
        predictor.data.add_prediction(dt, prediction.to_numpy())
        return prediction

    def _refresh_prediction_horizon(self, dt: datetime.datetime, knowledge: SensorKnowledge) -> None:
        predictor = knowledge.predictor
        predictor.update_prediction_horizon(dt)
        predictor.data.add_horizon_update(dt, predictor.model_metadata.uuid)
