import datetime
import logging

import pandas as pd

from sensor.sensor_knowledge import SensorKnowledge


def replace_current_model(dt: datetime.datetime, knowledge: SensorKnowledge) -> None:
    new_model = next(iter(knowledge.model_portfolio.get_models().values()))
    logging.info(
        f"{dt.isoformat()} SN Executor:  current model no longer in portfolio, replacing with {new_model.metadata.uuid}"
    )
    knowledge.predictor.set_model(new_model, dt)
    knowledge.predictor.data.add_configuration_update(dt, new_model.metadata.uuid)


def update_and_adjust_predictions(dt: datetime.datetime, knowledge: SensorKnowledge) -> None:
    if knowledge.predictor.model_metadata.uuid not in knowledge.model_portfolio.get_model_ids():
        replace_current_model(dt, knowledge)
    else:
        knowledge.update_prediction_horizon(dt)
    prediction: pd.Series = knowledge.predictor.get_prediction_at(dt)
    measurement: pd.Series = knowledge.predictor.data.get_measurements().loc[dt]
    knowledge.adjust_predictions(dt, measurement, prediction)


def notify_violation(dt: datetime.datetime, knowledge: SensorKnowledge) -> None:
    logging.info(f"{dt.isoformat()} SN Executor: notify violation to BS")
    predictor = knowledge.predictor
    measurement = knowledge.predictor.data.get_measurements().loc[dt]
    update = knowledge.base_station.notify_violation(dt, predictor, measurement)
    if update is not None:
        new_portfolio: set[str] = update.models_portfolio
        logging.info(f"{dt.isoformat()} SN Executor: new portfolio received {[model for model in new_portfolio]}")
        knowledge.update(dt, update)
        if knowledge.predictor.model_metadata.uuid not in new_portfolio:
            replace_current_model(dt, knowledge)
