import logging
from typing import Optional, Union

import pandas as pd
import tensorflow as tf
from keras.src.callbacks import History

from base.model import Model
from base.training import mse_weighted
from base.window_generator import WindowGenerator
from common.utils import convert_datetime, normalize_df, split_df


class ModelTrainer:

    def __init__(self):
        pass

    def retrain_model(self,
                      model: Model,
                      data: pd.DataFrame,
                      epochs: int,
                      patience: int,
                      optimizer: str,
                      learning_rate: float,
                      stride: int,
                      validation: Optional[Union[str, float]],
                      batch_size: int = 32,
                      ) -> (Model, History):
        """
        Retrains a model on all available data (initial + updates), using early stopping and train-validation split.

        If no validation argument is given, the model is trained the specified number of epochs without validation.
        On each training, a new model is created (with newly initialized weights) and trained on all collected data.
        The new model is always returned (not comparing its performance to the previous model).

        Args:
            model: The model to be retrained.
            data: The data to use for the retraining.
            epochs: Maximum number of epochs to train the model.
            patience: Number of epochs to wait for improvement of loss function before early stopping.
            optimizer: Optimizer to use for training.
            learning_rate: Learning rate for the optimizer.
            stride: Stride of the sliding window used for training.
            validation: Duration of last data to use for validation, e.g., '4w' for 4 weeks or 0.2 for 20% of the data.
            batch_size: The batch size used for training.
        """
        metadata = model.metadata
        convert_datetime(data, metadata.periodicity)
        if validation is None:
            train_df = data.copy()
            val_df = None
            cb = None
        else:
            cb = [
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     patience=int(patience / 2),
                                                     mode='min',
                                                     factor=0.2,
                                                     verbose=0
                                                     ),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)]
            if isinstance(validation, float):
                train_df, val_df, _ = split_df(data, 1 - validation, validation)
            else:  # validation is a timedelta
                train_df = data.loc[:data.index[-1] - validation, :].copy()
                val_df = data.loc[data.index[-1] - validation:, :].copy()

        # # adjust the normalization values to the new data
        norm_mean, norm_std = normalize_df(metadata.input_features, train_df, [] if val_df is None else [val_df])
        window = WindowGenerator(metadata.input_length,
                                 metadata.output_length,
                                 stride,
                                 sampling_rate=1,
                                 train_df=train_df, val_df=val_df, test_df=None,
                                 norm_mean=norm_mean, norm_std=norm_std,
                                 periodicity=metadata.periodicity,
                                 input_features=metadata.input_features,
                                 output_features=metadata.output_features,
                                 batch_size=batch_size
                                 )
        tf.keras.backend.clear_session()
        original_model = model.model
        new_model = tf.keras.models.clone_model(original_model)
        new_model.compile(
            optimizer=self._get_optimizer(optimizer, learning_rate),
            loss=mse_weighted,
            metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()]
        )
        history = new_model.fit(window.train, validation_data=window.val, epochs=epochs, callbacks=cb, verbose=1)
        new_md = metadata.deepcopy()
        new_md.input_normalization_mean = norm_mean
        new_md.input_normalization_std = norm_std
        return Model(new_model, new_md), history

    def fine_tune_model(self,
                        model: Model,
                        data: pd.DataFrame,
                        freeze_layers: bool,
                        epochs: int = 100,
                        stride: int = 1,
                        validation: Optional[str] = None,
                        ) -> Optional[Model]:
        """
        Applies transfer learning to the model by freezing the defined layers for training.

        New measurements are kept in a temporary buffer that is emptied after training a new model.
        The strategy starts with an empty buffer and a model that is trained on the initial data.
        If the freeze_layers list is empty, all weights remain trainable. Hence, a fine-tuning strategy is applied.

        Args:
            model: The model to be fine-tuned
            data: The data to use for the fine-tuning.
            freeze_layers: Freeze the core layers before training.
            epochs: Maximum number of epochs to train the model.
            stride: Stride of the sliding window used for training.
            validation: Duration of last data to use for validation, e.g., '4w' for 4 weeks.
        """
        md = model.metadata
        convert_datetime(data, md.periodicity)
        validation = None if validation is None else pd.Timedelta(validation)

        if validation is not None:
            train_df = data.loc[:data.index[-1] - validation, :].copy()
            val_df = data.loc[data.index[-1] - validation:, :].copy()
            md.apply_normalization(val_df)
            callbacks = [
                tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                     patience=int(epochs / 6),
                                                     mode='min',
                                                     factor=0.2,
                                                     verbose=0
                                                     ),
                tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=int(epochs / 4),
                                                 restore_best_weights=True
                                                 )]
        else:
            train_df = data.copy()
            val_df = None
            callbacks = None
        # we must not change the normalization parameters of the base model
        md.apply_normalization(train_df)

        window = WindowGenerator(md.input_length, md.output_length,
                                 stride=stride,
                                 sampling_rate=1,  # TODO: support sub-hourly data resolution
                                 train_df=train_df, val_df=val_df, test_df=None,
                                 norm_mean=md.input_normalization_mean, norm_std=md.input_normalization_std,
                                 periodicity=md.periodicity,
                                 input_features=md.input_features,
                                 output_features=md.output_features
                                 )
        # tf.keras.backend.clear_session()
        # new_model: tf.keras.Model = tf.keras.models.clone_model(model.model)
        # new_model.compile(
        #     optimizer=self._get_optimizer(optimizer, learning_rate),
        #     loss=mse_weighted,
        #     metrics=[tf.metrics.MeanSquaredError(), tf.metrics.MeanAbsoluteError(), tf.metrics.RootMeanSquaredError()]
        # )
        # new_model.set_weights(model.model.get_weights())

        # if freeze_layers:
        #     for layer in new_model.layers:
        #         # FIXME: the choice of layers to freeze shouldn't be hardcoded
        #         if layer.name == 'dense1':
        #             layer.trainable = False
        old_weights = model.model.get_weights()
        model.model.fit(window.train, validation_data=window.val, epochs=epochs, callbacks=callbacks, verbose=2)
        metadata = model.metadata.deepcopy()

        if validation is not None:
            new_weights = model.model.get_weights()
            new = model.model.evaluate(window.val)

            model.model.set_weights(old_weights)
            old = model.model.evaluate(window.val)

            logging.info(f"New model loss: {new[0]} - Old model loss: {old[0]}")

            if new[0] < old[0]:  # compare loss
                # found a better model, remove all except validation data
                model.model.set_weights(new_weights)
                return Model(model.model, metadata)
            else:
                logging.info('No improvement found, keeping old model.')
                return None

        return Model(model.model, metadata)

    @staticmethod
    def _get_optimizer(optimizer: str, learning_rate: float) -> tf.optimizers.Optimizer:
        if optimizer == 'adam':
            return tf.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer == 'rmsprop':
            return tf.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f'Unsupported optimizer: {optimizer}')
