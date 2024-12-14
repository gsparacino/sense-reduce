import datetime
from abc import ABC, abstractmethod
from typing import Optional

from base.cluster_configuration import ClusterConfiguration
from base.node_manager import NodeManager
from common.data_reduction_strategy import DataReductionStrategy


class ClusterAdaptationGoal(ABC):

    def __init__(self, goal_id: str):
        self.goal_id = goal_id

    @abstractmethod
    def is_violation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        pass

    @abstractmethod
    def requires_adaptation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        pass

    @abstractmethod
    def get_next_adaptation_dt(self, dt: datetime.datetime, node: NodeManager) -> Optional[datetime.datetime]:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @classmethod
    def from_dict(cls, data: dict) -> 'ClusterAdaptationGoal':
        goal_type = data['type']

        match goal_type:
            case DeployOnceAdaptationGoal.type_id:
                return DeployOnceAdaptationGoal.deserialize(data)
            case FixedIntervalAdaptationGoal.type_id:
                return FixedIntervalAdaptationGoal.deserialize(data)
            case _:
                raise ValueError(f'ClusterAdaptationGoal type {goal_type} not supported')

    @classmethod
    def evaluate_configuration_by_node_goals_score(
            cls,
            configuration: ClusterConfiguration,
            data_reduction_strategy: DataReductionStrategy,
            dt: datetime.datetime,
            node: NodeManager,
    ):
        configuration_models = configuration.models
        best_score = 0
        for model in configuration_models:
            predictor = Predictor(model, node.data)
            node_goals = node.get_adaptation_goals()
            model_score = (
                    sum(goal.evaluate(predictor, data_reduction_strategy, dt) for goal in node_goals) / len(node_goals))
            if model_score > best_score:
                best_score = model_score
        return best_score


class DeployOnceAdaptationGoal(ClusterAdaptationGoal):
    type_id: str = 'deploy_once'

    def __init__(self, goal_id: str, adaptation_cooldown_enabled: bool = False):
        super().__init__(goal_id)
        self._adaptation_cooldown_enabled = adaptation_cooldown_enabled

    def is_violation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        return False

    def requires_adaptation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        return False

    def get_next_adaptation_dt(self, dt: datetime.datetime, node: NodeManager) -> Optional[datetime.datetime]:
        return None if self._adaptation_cooldown_enabled else dt

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


class ContinuousDeploymentGoal(ClusterAdaptationGoal):
    type_id: str = 'continuous_deployment'

    def is_violation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        try:
            node.data.get_violations().loc[dt]
        except KeyError:
            return False
        return True

    def requires_adaptation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        return True

    def get_next_adaptation_dt(self, dt: datetime.datetime, node: NodeManager) -> Optional[datetime.datetime]:
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


class FixedIntervalAdaptationGoal(ClusterAdaptationGoal):
    type_id: str = 'fixed_interval'

    def __init__(self, goal_id: str, interval: datetime.timedelta, adaptation_cooldown: bool = False):
        super().__init__(goal_id)
        self.interval = interval
        self._adaptation_cooldown_enabled = adaptation_cooldown

    def is_violation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        try:
            node.data.get_violations().loc[dt]
        except KeyError:
            return False
        return True

    def requires_adaptation(self, dt: datetime.datetime, node: NodeManager) -> bool:
        last_adaptation_dt = node.last_adaptation_dt
        interval_expired = (dt - last_adaptation_dt) > self.interval
        return self.is_violation(dt, node) and interval_expired

    def get_next_adaptation_dt(self, dt: datetime.datetime, node: NodeManager) -> Optional[datetime.datetime]:
        return (dt + self.interval) if self._adaptation_cooldown_enabled else dt

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
