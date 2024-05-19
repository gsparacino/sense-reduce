import json
import os
from typing import Any

import tensorflow as tf

from base.model import Model
from common import ModelMetadata, PredictionModel, LiteModel


def load_model(path: os.path) -> Model:
    """
    Loads the model from the specified directory, assuming it contains a 'metadata.json' file.

    :param path: the path to the directory containing the model
    :return: the loaded model
    """
    metadata_path = os.path.join(path, ModelMetadata.FILE_NAME)
    metadata = _load_model_metadata(str(metadata_path))
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
    with open(path, 'r') as f:
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
    _save_model_metadata(model.metadata, path)
    # Save model's bytes in TFLite format
    model_bytes: bytes = _to_tflite_model_bytes(model.model)
    save_model_as_tflite(model_bytes, path, model.model_id)


def save_model_as_tflite(model_bytes: bytes, path: os.path, model_name: str) -> None:
    """
    Saves the model in TFLite format, and its metadata in JSON format.

    :param model_bytes: the model to save, as a bytes string
    :param path: the path to save the model to save
    :param model_name: the name of the model to save
    """
    os.makedirs(path, exist_ok=True)
    model_path = _get_model_path(path, model_name)
    with open(model_path, 'wb') as f:
        f.write(model_bytes)


def load_model_from_tflite(path: os.path, model_name: str, model_metadata: ModelMetadata) -> PredictionModel:
    """
    Loads a model from a TFLite file.

    :param path: the path to load the model from
    :param model_name: the ID of the model to load
    :param model_metadata: the metadata of the model to load
    :return: the loaded PredictionModel
    """
    model_path = _get_model_path(path, model_name)
    return LiteModel.from_tflite_file(model_path, model_metadata)


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


def _get_model_path(model_dir: str, model_name: str):
    file_name = f'{model_name}.tflite'
    model_path = os.path.join(model_dir, file_name)
    return model_path
