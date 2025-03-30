import datetime
import logging
from typing import Callable

from base.base_station_configuration import BaseStationConfiguration
from base.base_station_knowledge import BaseStationKnowledge


def update_node_configuration(
        node_id: str,
        configuration: BaseStationConfiguration
) -> Callable[[datetime.datetime, BaseStationKnowledge], None]:
    def f(
            dt: datetime.datetime,
            knowledge: BaseStationKnowledge
    ) -> None:
        logging.info(
            f"{dt.isoformat()} BS Executor: updating {node_id} models: {[model.metadata.uuid for model in configuration.models]}")
        node_manager = knowledge.get_node(node_id)
        configuration.apply(dt, node_manager)
        node_manager.predictor.add_configuration_update(dt, configuration.option_id)

    return f


def update_last_adaptation_dt(
        node_id: str,
) -> Callable[[datetime.datetime, BaseStationKnowledge], None]:
    def f(
            dt: datetime.datetime,
            knowledge: BaseStationKnowledge
    ) -> None:
        logging.info(f"{dt.isoformat()} BS Executor: updating {node_id} last_adaptation_dt")
        node_manager = knowledge.get_node(node_id)
        node_manager.last_adaptation_dt = dt
        node_manager.next_sync_dt = node_manager.next_sync_dt

    return f
