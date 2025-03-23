import datetime
import logging
from typing import Optional

import pandas as pd

from common.data_reduction_strategy import DataReductionStrategy
from common.predictor import Predictor
from common.sensor_knowledge_update import SensorKnowledgeUpdate, SensorKnowledgeInitialization
from sensor.base_station_gateway import BaseStationGateway


class BaseStationAdapter:

    def __init__(self, gateway: BaseStationGateway, data_reduction_strategy: DataReductionStrategy):
        self.gateway = gateway
        self.data_reduction_strategy = data_reduction_strategy

    def register_node(self,
                      input_features: list[str],
                      output_features: list[str]
                      ) -> SensorKnowledgeInitialization:
        return self.gateway.register_node(input_features, output_features)

    def synchronize(self, dt: datetime.datetime) -> SensorKnowledgeUpdate:
        return self.gateway.synchronize(dt)

    def notify_violation(
            self,
            dt: datetime.datetime,
            predictor: Predictor,
            measurement: pd.Series
    ) -> Optional[SensorKnowledgeUpdate]:
        # next_sync_dt = predictor.data.next_synchronization_dt
        # if next_sync_dt and next_sync_dt <= dt:
        data = self.data_reduction_strategy.get_violation_data(predictor.data, predictor.model_metadata, dt)
        if data is None:
            logging.info(f"{dt.isoformat()} SN: sending violation without data")
        else:
            logging.info(
                f"{dt.isoformat()} SN: Sending violation with measurements: {[ts.isoformat() for ts in data.index]}"
            )
        configuration_id = predictor.model_metadata.uuid
        result = self.gateway.send_violation(dt, configuration_id, measurement, data)
        predictor.data.last_synchronization_dt = dt
        return result

    # return None

    def notify_horizon_update(
            self,
            dt: datetime.datetime,
            predictor: Predictor
    ) -> Optional[SensorKnowledgeUpdate]:
        next_sync_dt = predictor.data.next_synchronization_dt
        if next_sync_dt and next_sync_dt <= dt:
            data = self.data_reduction_strategy.get_horizon_update_data(predictor.data, predictor.model_metadata, dt)
            if data is None or data.size == 0:
                logging.info(f"{dt.isoformat()} SN: Sending horizon update without measurements")
            else:
                logging.info(
                    f"{dt.isoformat()} SN: Sending horizon update with measurements {[ts.isoformat() for ts in data.index]}"
                )
            configuration_id = predictor.model_metadata.uuid
            result = self.gateway.send_horizon_update(dt, data, configuration_id)
            predictor.data.last_synchronization_dt = dt
            return result
        return None
