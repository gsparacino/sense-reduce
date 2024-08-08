import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import keras
import pandas as pd
import tensorflow as tf

from base import Config
from base.model import ModelID, Model
from base.training import mse_weighted
from base.window_generator import WindowGenerator
from common import ModelMetadata, normalize_df, split_df, convert_datetime


class LearningStrategy(ABC):
    """
    An interface for defining strategies for continual learning of machine learning models.

    By subclassing this class, new strategies can be implemented. Strategies for retraining and transfer learning are
    already implemented. In the future, strategies for other continual learning methods will be added.
    """

    @abstractmethod
    def train_new_model(self, old_model: keras.Model, metadata: ModelMetadata, data: pd.DataFrame) -> Model:
        pass


class NoUpdateLearningStrategy(LearningStrategy):
    """A LearningStrategy that does not execute any update."""

    def train_new_model(self, old_model: keras.Model, metadata: ModelMetadata, data: pd.DataFrame) -> Model:
        return Model(old_model, metadata)


class RetrainLearningStrategy(LearningStrategy):
    """
    A LearningStrategy that retrains a model on all available data, using early stopping and train-validation split.

   If no validation argument is given, the model is trained the specified number of epochs without validation.
   On each training, a new model is created (with newly initialized weights) and trained on all collected data.
   The new model is always returned (not comparing its performance to the previous model).

   Args:
       epochs: Maximum number of epochs to train the model.
       patience: Number of epochs to wait for improvement of loss function before early stopping.
       optimizer: Optimizer to use for training.
       learning_rate: Learning rate for the optimizer.
       stride: Stride of the sliding window used for training.
       validation: Duration of last data to use for validation, e.g., '4w' for 4 weeks or 0.2 for 20% of the data.
   """

    def __init__(self,
                 epochs: int = 100,
                 patience: int = 20,
                 optimizer: str = 'adam',
                 learning_rate: float = 0.001,
                 stride: int = 1,
                 validation: Optional[Union[str, float]] = None,
                 ):
        self._models: Dict[ModelID, Model] = {}
        self.epochs = epochs
        self.patience = patience
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.stride = stride
        self.validation = validation

    def train_new_model(self, old_model: keras.Model, metadata: ModelMetadata, data: pd.DataFrame) -> Model:
        convert_datetime(data, metadata.periodicity)
        if self.validation is None:
            train_df = data.copy()
            val_df = None
            cb = None
        else:
            cb = [
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     patience=int(self.patience / 2),
                                                     mode='min',
                                                     factor=0.2,
                                                     verbose=1
                                                     ),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=self.patience, restore_best_weights=True)]
            if isinstance(self.validation, float):
                train_df, val_df, _ = split_df(data, 1 - self.validation, self.validation)
            else:  # self.validation is a timedelta
                train_df = data.loc[:data.index[-1] - self.validation, :].copy()
                val_df = data.loc[data.index[-1] - self.validation:, :].copy()

        # # adjust the normalization values to the new data
        norm_mean, norm_std = normalize_df(metadata.input_features, train_df, [] if val_df is None else [val_df])
        window = WindowGenerator(metadata.input_length, metadata.output_length, self.stride,
                                 sampling_rate=1,
                                 train_df=train_df, val_df=val_df, test_df=None,
                                 norm_mean=norm_mean, norm_std=norm_std,
                                 periodicity=metadata.periodicity,
                                 input_features=metadata.input_features,
                                 output_features=metadata.output_features
                                 )
        tf.keras.backend.clear_session()
        new_model = tf.keras.models.clone_model(old_model)
        new_model.compile(
            optimizer=self._get_optimizer(),
            loss=mse_weighted,
            metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()]
        )
        new_model.fit(window.train, validation_data=window.val, epochs=self.epochs, callbacks=cb, verbose=2)
        metadata = metadata.deepcopy()
        metadata.input_normalization_mean = norm_mean
        metadata.input_normalization_std = norm_std
        metadata.model_id = str(uuid.uuid4())
        return Model(new_model, metadata)

    def _get_optimizer(self) -> tf.optimizers.Optimizer:
        if self.optimizer == 'adam':
            return tf.optimizers.Adam(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            return tf.optimizers.RMSprop(learning_rate=self.learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer: {self.optimizer}')


def learning_strategy_factory(config: Config) -> LearningStrategy:
    if config.learning_strategy:
        match config.learning_strategy.lower():
            case "retrain":
                logging.debug("Using RetrainLearningStrategy as LearningStrategy")
                return RetrainLearningStrategy()
            case "noupdate":
                logging.debug("Using NoUpdateLearningStrategy as LearningStrategy")
                return NoUpdateLearningStrategy()
            case _:
                logging.debug(
                    "No strategy matches config.yaml's parameter learning_strategy, using default "
                    "(NoUpdateLearningStrategy)"
                )
                return NoUpdateLearningStrategy()
    logging.debug("Missing parameter learning_strategy in config.yaml, using default (NoUpdateLearningStrategy)")
