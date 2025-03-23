import datetime
from abc import abstractmethod

from base.model import Model
from base.node_manager import NodeManager


class BaseStationConfiguration:

    def __init__(self, option_id: str, models: set[Model], next_sync_dt: datetime.datetime):
        self.option_id = option_id
        self.models = models
        self.next_sync_dt = next_sync_dt

    def is_active(self, node: NodeManager) -> bool:
        return node.get_model_ids() == self.models

    @abstractmethod
    def apply(self, dt: datetime.datetime, node: NodeManager) -> None:
        model_ids = set([model.metadata.uuid for model in self.models])
        node.set_models(model_ids)
        node.last_adaptation_dt = dt
        node.next_sync_dt = self.next_sync_dt
