import datetime
from abc import ABC, abstractmethod
from typing import Optional

from base.node_manager import NodeManager


class PortfolioAdaptationGoal(ABC):

    def __init__(self, goal_id: str):
        self.goal_id = goal_id

    @abstractmethod
    def is_violation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        pass

    @abstractmethod
    def requires_adaptation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        pass

    @abstractmethod
    def get_next_update_dt(self, dt: datetime.datetime, node: NodeManager) -> Optional[datetime.datetime]:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @classmethod
    def from_dict(cls, data: dict) -> 'PortfolioAdaptationGoal':
        goal_type = data['type']

        match goal_type:
            case DeployOnceAdaptationGoal.type_id:
                return DeployOnceAdaptationGoal.deserialize(data)
            case FixedIntervalAdaptationGoal.type_id:
                return FixedIntervalAdaptationGoal.deserialize(data)
            case _:
                raise ValueError(f'ClusterAdaptationGoal type {goal_type} not supported')


class DeployOnceAdaptationGoal(PortfolioAdaptationGoal):
    type_id: str = 'deploy_once'

    def __init__(self, goal_id: str):
        super().__init__(goal_id)

    def is_violation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        return False

    def requires_adaptation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        return False

    def get_next_update_dt(self, dt: datetime.datetime, node: NodeManager) -> Optional[datetime.datetime]:
        return dt

    def to_dict(self) -> dict:
        return {
            'type': DeployOnceAdaptationGoal.type_id,
            'goal_id': self.goal_id,
        }

    @classmethod
    def deserialize(cls, data: dict) -> 'DeployOnceAdaptationGoal':
        if data['type'] != DeployOnceAdaptationGoal.type_id:
            raise ValueError(f"Invalid type id {data['type']} for {cls.__name__}")

        return DeployOnceAdaptationGoal(
            data['goal_id']
        )


class FixedIntervalAdaptationGoal(PortfolioAdaptationGoal):
    type_id: str = 'fixed_interval'

    def __init__(self, goal_id: str, interval: datetime.timedelta):
        super().__init__(goal_id)
        self.interval = interval

    def is_violation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        return self._update_interval_expired(dt, node)

    def requires_adaptation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        return self._update_interval_expired(dt, node)

    def _update_interval_expired(self, dt: datetime.datetime, node: NodeManager) -> bool:
        last_adaptation_dt = node.last_adaptation_dt
        return (dt - last_adaptation_dt) > self.interval

    def get_next_update_dt(self, dt: datetime.datetime, node: NodeManager) -> Optional[datetime.datetime]:
        return (dt + self.interval) if self.interval else dt

    def to_dict(self) -> dict:
        return {
            'type': FixedIntervalAdaptationGoal.type_id,
            'goal_id': self.goal_id,
            'interval_seconds': self.interval.total_seconds(),
        }

    @classmethod
    def deserialize(cls, data: dict) -> 'FixedIntervalAdaptationGoal':
        if data['type'] != FixedIntervalAdaptationGoal.type_id:
            raise ValueError(f"Invalid type id {data['type']} for {cls.__name__}")

        return FixedIntervalAdaptationGoal(
            data['goal_id'],
            datetime.timedelta(seconds=data['interval_seconds']),
        )
