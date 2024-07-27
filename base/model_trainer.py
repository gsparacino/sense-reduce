import uuid
from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import keras
import pandas as pd
import tensorflow as tf

from base.model import ModelID, Model
from base.training import mse_weighted
from base.window_generator import WindowGenerator
from common import ModelMetadata, normalize_df, split_df, convert_datetime


class LearningStrategy(ABC):

    @abstractmethod
    def train_new_model(self, old_model: keras.Model, metadata: ModelMetadata, data: pd.DataFrame) -> Model:
        pass


class RetrainLearningStrategy(LearningStrategy):

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
