import datetime
import logging
from typing import Optional

import pandas as pd

from common.data_reduction_strategy import DataReductionStrategy
from common.predictor import Predictor
from common.sensor_knowledge_update import SensorKnowledgeUpdate, SensorKnowledgeInitialization
from sensor.base_station_gateway import BaseStationGateway


class BaseStation:

    def __init__(self, gateway: BaseStationGateway, data_reduction_strategy: DataReductionStrategy):
        self.gateway = gateway
        self.data_reduction_strategy = data_reduction_strategy

    def register_node(self,
                      initial_df: pd.DataFrame,
                      input_features: list[str],
                      output_features: list[str]
                      ) -> SensorKnowledgeInitialization:
        return self.gateway.register_node(initial_df, input_features, output_features)

    def sync(self, dt: datetime.datetime) -> SensorKnowledgeUpdate:
        return self.gateway.sync(dt)

    def on_violation(
            self,
            dt: datetime.datetime,
            predictor: Predictor,
            measurement: pd.Series
    ) -> Optional[SensorKnowledgeUpdate]:
        if self.data_reduction_strategy.synchronization_enabled(predictor.data, dt):
            data = self.data_reduction_strategy.on_violation(predictor.data, predictor.model_metadata, dt)
            if data is None:
                logging.info(f"{dt.isoformat()} SN: sending violation without data")
            else:
                logging.info(
                    f"{dt.isoformat()} SN: Sending violation with measurements: {[ts.isoformat() for ts in data.index]}"
                )
            configuration_id = predictor.model_metadata.uuid
            result = self.gateway.send_violation(dt, configuration_id, measurement, data)
            predictor.data.last_synchronization_dt = dt
            predictor.data.next_synchronization_dt = result.next_adaptation_dt
            return result
        return None

    def on_horizon_update(
            self,
            dt: datetime.datetime,
            predictor: Predictor
    ) -> Optional[SensorKnowledgeUpdate]:
        if self.data_reduction_strategy.synchronization_enabled(predictor.data, dt):
            data = self.data_reduction_strategy.on_horizon_update(predictor.data, predictor.model_metadata, dt)
            if data is None or data.size == 0:
                logging.info(f"{dt.isoformat()} SN: Sending horizon update without measurements")
            else:
                logging.info(
                    f"{dt.isoformat()} SN: Sending horizon update with measurements {[ts.isoformat() for ts in data.index]}"
                )
            configuration_id = predictor.model_metadata.uuid
            result = self.gateway.send_update(dt, data, configuration_id)
            predictor.data.last_synchronization_dt = dt
            predictor.data.next_synchronization_dt = result.next_adaptation_dt
            return result
        return None
