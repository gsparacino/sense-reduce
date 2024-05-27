import datetime
import json
import logging
import os
import shutil
from typing import Any, Optional

import numpy as np
import tensorflow as tf

from base.model import Model
from common import ModelMetadata, PredictionModel, LiteModel, ThresholdMetric, Predictor


def load_model_from_savemodel(path: os.path) -> Model:
    """
    Loads the model from the specified directory, assuming it contains a 'metadata.json' file and model's file in a
    SaveModel format.

    :param path: the path to the directory containing the model in SaveModel format
    :return: the loaded model
    """
    metadata = _load_model_metadata(path)
    keras_model = tf.keras.models.load_model(path)
    return Model(keras_model, metadata)


def _save_model_metadata(metadata: ModelMetadata, path: str) -> None:
    """
    Saves the model metadata in JSON format in the specified directory.

    :param metadata: the model metadata
    :param path: the directory to save the metadata in
    """
    with open(os.path.join(path, ModelMetadata.FILE_NAME), 'w') as f:
        json.dump(metadata.to_dict(), f, separators=(',', ':'))


def _load_model_metadata(path: str) -> ModelMetadata:
    """
    Loads the model metadata from the specified directory, assuming it contains a 'metadata.json' file.

    :param path: the path to the metadata file
    :return: a ModelMetadata instance
    """
    file_path = os.path.join(path, ModelMetadata.FILE_NAME)
    with open(file_path, 'r') as f:
        file = json.load(f)
        return ModelMetadata.from_dict(file)


def _to_tflite_model_bytes(model: tf.keras.Model) -> Any:
    """Creates a TFLite model from this model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    return converter.convert()


def save_model(model: Model, path: os.path) -> None:
    """
    Saves the model in SavedModel and TFLite format, and its metadata in JSON format.

    :param model: the model to save
    :param path: the path to save the model to
    """
    os.makedirs(path, exist_ok=True)
    # Save model in SaveModel's format
    keras_model = model.model
    keras_model.save(path)
    # Save model's bytes in TFLite format
    model_bytes: bytes = _to_tflite_model_bytes(model.model)
    save_model_as_tflite(model_bytes, model.metadata, path)


def save_model_as_tflite(model_bytes: bytes, metadata: ModelMetadata, path: os.path) -> None:
    """
    Saves the model in TFLite format, and its metadata in JSON format.

    :param model_bytes: The model to save, as a bytes string
    :param metadata: The metadata of the model to save
    :param path: The path to save the model to
    """
    os.makedirs(path, exist_ok=True)
    _save_model_metadata(metadata, path)
    model_path = get_model_tflite_path(path, metadata.model_id)
    with open(model_path, 'wb') as f:
        f.write(model_bytes)


def load_model_tflite(model_dir: str) -> PredictionModel:
    if not os.path.isdir(model_dir):
        raise NotADirectoryError(f'Model directory {model_dir} does not exist')
    metadata = _load_model_metadata(model_dir)
    path = get_model_tflite_path(model_dir, metadata.model_id)
    model = LiteModel.from_tflite_file(path, metadata)
    return model


def delete_tflite_model(model_dir: str) -> None:
    if os.path.isdir(model_dir):
        shutil.rmtree(model_dir)


def clone_model(model: Model) -> Model:
    """
    Creates an identical copy of a model.

    :param model: the model to clone
    :return: the cloned model
    """
    clone = tf.keras.models.clone_model(model.model)
    clone.set_weights(model.model.get_weights())
    metadata = model.metadata.deepcopy()
    return Model(clone, metadata)


def get_model_tflite_path(model_dir: str, model_name: str):
    """
    Returns the full path of a model's .tflite file.

    :param model_dir: The directory containing the model's files
    :param model_name: The name of the model
    :return: The full path of the model's .tflite file, as a string
    """
    file_name = _get_model_tflite_file_name(model_name)
    model_path = os.path.join(model_dir, file_name)
    return model_path


def _get_model_tflite_file_name(model_name: str) -> str:
    return f'{model_name}.tflite'


def get_model_dir_path(models_root_folder: str, model_name: str):
    """
    Returns the path of the directory containing the model's files.

    :param models_root_folder: The 'root model folder', i.e., the folder containing all the models
    :param model_name: The unique name of the model
    :return: The full path of the model's folder, as a string
    """
    return os.path.join(models_root_folder, model_name)


def get_better_predictor(models: list[PredictionModel],
                         threshold_metric: ThresholdMetric,
                         current_predictor: Predictor,
                         timestamp: datetime.datetime,
                         measurements: np.array,
                         prediction: np.array) -> Optional[Predictor]:
    """
    Iterates over the provided PredictionModels, comparing their performance on the latest measurements and returning
    a Predictor with the best model.

    :param models: the list of models to evaluate against the threshold metric
    :param threshold_metric: the threshold metric used to rank the models
    :param current_predictor: the Predictor currently used by the Sensor
    :param timestamp: the timestamp when the latest measurements were taken
    :param measurements: the latest measurements
    :param prediction: the latest prediction

    :return: a Predictor with the best possible PredictionModel according to the threshold_metric, or None if none
    of the models has better performances than the current one.
    """
    best_score = threshold_metric.threshold_score(measurements, prediction)
    best_predictor = None
    for model in models:
        if model.metadata.model_id == current_predictor.model_id:
            continue
        predictor = Predictor(model, current_predictor.data, current_predictor.prediction_period)
        predictor.update_prediction_horizon(timestamp)
        prediction = predictor.get_prediction_at(timestamp).to_numpy()
        score = threshold_metric.threshold_score(measurements, prediction)
        logging.debug(f"{predictor.model_id} score: {best_score}")
        if score < best_score:
            best_score = score
            best_predictor = predictor
            logging.debug(f"New best model: {predictor.model_id}")
    return best_predictor
