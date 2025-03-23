import datetime
import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from base.model import Model
from base.model_manager import ModelManager
from base.model_trainer import ModelTrainer
from base.node_manager import NodeManager


class LearningStrategy(ABC):

    def __init__(self, model_trainer: ModelTrainer, model_manager: ModelManager):
        self.model_trainer = model_trainer
        self.model_manager = model_manager

    @abstractmethod
    def get_candidate_models(self, node: NodeManager, dt: datetime.datetime) -> set[Model]:
        pass


class NoUpdateStrategy(LearningStrategy):

    def __init__(self, model_trainer: ModelTrainer, model_manager: ModelManager):
        super().__init__(model_trainer, model_manager)

    def get_candidate_models(self, node: NodeManager, dt: datetime.datetime) -> set[Model]:
        return set()


class RetrainStrategy(LearningStrategy):

    def __init__(self, model_trainer: ModelTrainer, model_manager: ModelManager):
        super().__init__(model_trainer, model_manager)

    def get_candidate_models(self, node: NodeManager, dt: datetime.datetime) -> set[Model]:
        model_manager = self.model_manager

        active_model_id = node.get_active_model_id()
        model_to_retrain = model_manager.get_model(active_model_id)
        measurements = node.predictor.get_measurements()
        new_model = self.retrain_model(dt, model_to_retrain, measurements)
        model_manager.save_model(new_model)
        new_model_id = new_model.metadata.uuid
        logging.info(f"New model trained: {new_model_id}")
        new_portfolio = set([model_id for model_id in node.get_model_ids() if model_id != active_model_id])
        new_portfolio.add(new_model_id)
        return model_manager.get_models(new_portfolio)

    def retrain_model(self, dt: datetime.datetime, model: Model, measurements: pd.DataFrame) -> Model:
        model_id = model.metadata.uuid
        logging.info(f'Retraining model {model_id} with data {measurements.index.min()} - {measurements.index.max()}')
        new_model_id = f"{model_id}_{dt.strftime('%Y%m%d%H%M')}"
        # TODO: make these parameters configurable
        new_model, _ = self.model_trainer.retrain_model(
            model, measurements,
            epochs=100, patience=20, validation=0.3,
            stride=1, optimizer='rmsprop', learning_rate=0.002399431372613329,
        )
        new_model.metadata.uuid = new_model_id
        return new_model


class FineTuneStrategy(LearningStrategy):

    def __init__(self, model_trainer: ModelTrainer, model_manager: ModelManager):
        super().__init__(model_trainer, model_manager)

    def get_candidate_models(self, node: NodeManager, dt: datetime.datetime) -> set[Model]:
        model_id = node.get_active_model_id()
        model_to_retrain = self.model_manager.get_model(model_id)
        measurements = node.predictor.get_measurements()
        # FIXME
        new_model = self._fine_tune_model(dt, model_to_retrain, measurements, False, '4w')
        if new_model is not None:
            self.model_manager.save_model(new_model)
            new_model_id = new_model.metadata.uuid
            logging.info(f"New model trained: {new_model_id}")
            new_portfolio = set([m for m in node.get_model_ids() if m != model_id])
            new_portfolio.add(new_model_id)
            return self.model_manager.get_models(new_portfolio)
        else:
            return self.model_manager.get_models(node.get_model_ids())

    def _fine_tune_model(self,
                         dt: datetime.datetime,
                         model: Model,
                         measurements: pd.DataFrame,
                         freeze_layers: bool,
                         validation: str
                         ) -> Optional[Model]:
        model_id = model.metadata.uuid
        logging.info(f'Fine-tuning model {model_id}')
        new_model = self.model_trainer.fine_tune_model(
            model, measurements, freeze_layers=freeze_layers, validation=validation
        )
        if new_model is not None:
            new_model_id = f"{model_id}_{dt.strftime('%Y%m%d%H%M')}"
            new_model.metadata.uuid = new_model_id
        return new_model
