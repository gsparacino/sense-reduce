import datetime
import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from base.dense_model_trainer import DenseModelTrainer
from base.model import Model
from base.model_manager import ModelManager
from base.node_manager import NodeManager


class LearningStrategy(ABC):

    def __init__(self, model_trainer: DenseModelTrainer, model_manager: ModelManager):
        self.model_trainer = model_trainer
        self.model_manager = model_manager

    @abstractmethod
    def update_portfolio(self, node: NodeManager, dt: datetime.datetime) -> set[Model]:
        pass


class NoUpdateStrategy(LearningStrategy):

    def __init__(self, model_trainer: DenseModelTrainer, model_manager: ModelManager):
        super().__init__(model_trainer, model_manager)

    def update_portfolio(self, node: NodeManager, dt: datetime.datetime) -> set[Model]:
        return self.model_manager.get_models(node.get_model_ids())


class RetrainStrategy(LearningStrategy):

    def __init__(self, model_trainer: DenseModelTrainer, model_manager: ModelManager):
        super().__init__(model_trainer, model_manager)

    def update_portfolio(self, node: NodeManager, dt: datetime.datetime) -> set[Model]:
        model_id = node.get_active_model_id()
        model_to_retrain = self.model_manager.get_model(model_id)
        measurements = node.data.get_measurements()
        new_model = self._retrain_model(dt, model_to_retrain, measurements)
        self.model_manager.save_model(new_model)
        self.model_manager.delete_model(model_id)
        new_model_id = new_model.metadata.uuid
        logging.info(f"New model trained: {new_model_id}")
        new_portfolio = set([m for m in node.get_model_ids() if m != model_id])
        new_portfolio.add(new_model_id)
        return self.model_manager.get_models(new_portfolio)

    def _retrain_model(self, dt: datetime.datetime, model: Model, measurements: pd.DataFrame) -> Model:
        model_id = model.metadata.uuid
        logging.info(f'Retraining model {model_id} with data {measurements.index.min()} - {measurements.index.max()}')
        new_model_id = f"{model_id}_{dt.strftime('%Y%m%d%H%M')}"
        # TODO: make these parameters configurable
        new_model = self.model_trainer.retrain_model(
            model, measurements,
            epochs=100, patience=20, validation=0.3,
            stride=1, optimizer='rmsprop', learning_rate=0.002399431372613329,
        )
        new_model.metadata.uuid = new_model_id
        return new_model


class FineTuneStrategy(LearningStrategy):

    def __init__(self, model_trainer: DenseModelTrainer, model_manager: ModelManager):
        super().__init__(model_trainer, model_manager)

    def update_portfolio(self, node: NodeManager, dt: datetime.datetime) -> set[Model]:
        model_id = node.get_active_model_id()
        model_to_retrain = self.model_manager.get_model(model_id)
        measurements = node.data.get_measurements()
        # FIXME
        new_model = self._fine_tune_model(dt, model_to_retrain, measurements, False, '4w')
        self.model_manager.save_model(new_model)
        self.model_manager.delete_model(model_id)
        new_model_id = new_model.metadata.uuid
        logging.info(f"New model trained: {new_model_id}")
        new_portfolio = set([m for m in node.get_model_ids() if m != model_id])
        new_portfolio.add(new_model_id)
        return self.model_manager.get_models(new_portfolio)

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
