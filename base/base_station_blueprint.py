import datetime
import logging
from datetime import datetime

import pandas as pd
from flask import Blueprint, request, send_file, current_app

from base.base_station_manager import BaseStationManager
from base.model_manager import ModelManager
from common.sensor_knowledge_update import SensorKnowledgeInitialization, SensorKnowledgeUpdate

base_station_bp = Blueprint('base_station_blueprint', __name__)


@base_station_bp.post("/nodes")
def register_node():
    """Registers a new node and returns the initial configuration for the node."""
    body = request.get_json(force=True)
    input_features: list[str] = body['input_features']
    output_features: list[str] = body['output_features']
    initial_df_json = body.get('initial_df')
    initial_df = pd.read_json(initial_df_json) if initial_df_json is not None else None

    base_station_manager: BaseStationManager = current_app.config['BASE_STATION']
    assert base_station_manager is not None

    initialization: SensorKnowledgeInitialization = base_station_manager.register_node(
        input_features=input_features,
        output_features=output_features,
        initial_df=initial_df
    )

    logging.info(f'New node registration with ID={initialization.node_id}')
    return initialization.to_dict()


@base_station_bp.post("/nodes/<string:node_id>/violation")
def post_violation(node_id: str):
    body = request.get_json(force=True)
    timestamp: str = body['timestamp']
    measurements: str = body['measurement']
    configuration_id: str = body['configuration_id']
    data: str = body['data']

    logging.info(f'Received violation message from node {node_id} @ {timestamp}')

    dt = datetime.fromisoformat(timestamp)

    measurement_series: pd.Series = pd.read_json(measurements, typ='series', orient='records')
    data_df: pd.DataFrame = pd.read_json(data, typ='frame', orient='records')

    base_station_manager: BaseStationManager = current_app.config['BASE_STATION']
    assert base_station_manager is not None

    result: SensorKnowledgeUpdate = base_station_manager.handle_violation(
        node_id=node_id,
        dt=dt,
        measurement=measurement_series,
        data=data_df,
        configuration_id=configuration_id
    )

    payload = dict()
    payload['adaptation_goals'] = [goal.to_dict() for goal in result.predictor_adaptation_goals]
    payload['models_portfolio'] = list(result.models_portfolio)
    payload['next_sync_dt'] = result.next_sync_dt.isoformat()
    return payload


@base_station_bp.post("/nodes/<string:node_id>/update")
def post_update(node_id: str):
    """Called when a sensor node reaches the end of its prediction horizon."""
    body = request.get_json(force=True)
    dt: datetime = datetime.fromisoformat(body['timestamp'])
    data: pd.DataFrame = pd.read_json(body['data'])
    configuration_id: str = body['configuration_id']

    logging.info(f'Received update message from node {node_id} @ {dt.isoformat()}')

    base_station_manager: BaseStationManager = current_app.config['BASE_STATION']
    assert base_station_manager is not None

    result: SensorKnowledgeUpdate = base_station_manager.handle_horizon_update(
        node_id=node_id,
        dt=dt,
        data=data,
        configuration_id=configuration_id
    )

    payload = dict()
    payload['portfolio'] = [goal.to_dict() for goal in result.predictor_adaptation_goals]
    payload['adaptation_goals'] = list(result.models_portfolio)
    payload['next_sync_dt'] = result.next_sync_dt.isoformat()
    return payload


@base_station_bp.post("/nodes/<string:node_id>/measurement")
def post_measurement(node_id: str):
    """Endpoint for a node to communicate a measurement."""
    body = request.get_json(force=True)
    dt = datetime.datetime.fromisoformat(body['timestamp'])
    logging.info(f'Node {node_id} sent a measurement @ {dt.isoformat()}')
    # TODO: implement actual MAPEK logic on BS
    payload = dict()
    payload['portfolio'] = []  # TODO
    payload['adaptation_goals'] = []  # TODO
    return payload


@base_station_bp.get("/nodes/<string:node_id>/models/<string:model_id>")
def get_model(node_id: str, model_id: str):
    logging.info(f'Node {node_id} requested model {model_id}')
    model_manager: ModelManager = current_app.config['MODEL_MANAGER']
    assert model_manager is not None
    model_file_path = model_manager.get_model_tflite_file_path(model_id)
    return send_file(model_file_path)


@base_station_bp.get("/nodes/<string:node_id>/synchronize")
def synchronize(node_id: str):
    logging.info(f'Node {node_id} requested synchronization')

    base_station_manager: BaseStationManager = current_app.config['BASE_STATION']
    assert base_station_manager is not None

    result = base_station_manager.sync(node_id)

    payload = dict()
    payload['portfolio'] = [goal.to_dict() for goal in result.predictor_adaptation_goals]
    payload['adaptation_goals'] = list(result.models_portfolio)
    return payload


@base_station_bp.get("/nodes/<string:node_id>/models/<string:model_id>/metadata")
def get_model_metadata(node_id: str, model_id: str):
    logging.info(f'Node {node_id} requested model metadata of {model_id}')

    model_manager: ModelManager = current_app.config['MODEL_MANAGER']
    assert model_manager is not None

    metadata = model_manager.get_model(model_id).metadata
    return metadata.to_dict()


@base_station_bp.get("/ping")
def ping():
    """Simple GET endpoint for health checks."""
    logging.info(f'GET /ping, request: {request}')
    return "pong"
