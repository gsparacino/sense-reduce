import datetime

from common.predictor import Predictor
from sensor.sensor_knowledge import SensorKnowledge


class PredictorConfiguration:

    def __init__(self, option_id: str, predictor: Predictor):
        self.option_id = option_id
        self.predictor = predictor

    def is_active(self, knowledge: SensorKnowledge) -> bool:
        return knowledge.predictor.model_metadata.uuid == self.predictor.model_metadata.uuid

    def apply(self, dt: datetime.datetime, knowledge: SensorKnowledge) -> None:
        self.predictor._data = knowledge.predictor.data
        knowledge.predictor = self.predictor
        knowledge.predictor.data.add_configuration_update(dt, self.option_id)
