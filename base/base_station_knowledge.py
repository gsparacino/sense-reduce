import pandas as pd

from base.learning_strategy import LearningStrategy
from base.model_manager import ModelManager
from base.node_manager import NodeManager
from base.portfolio_adaptation_goal import PortfolioAdaptationGoal
from common.data_reduction_strategy import DataReductionStrategy
from common.predictor_adaptation_goal import PredictorAdaptationGoal
from common.sensor_knowledge_update import SensorKnowledgeInitialization


class BaseStationKnowledge:

    def __init__(
            self,
            base_station_id: str,
            model_manager: ModelManager,
            initial_df: pd.DataFrame,
            learning_strategy: LearningStrategy,
            data_reduction_strategy: DataReductionStrategy,
            cluster_adaptation_goals: list[PortfolioAdaptationGoal],
            sensor_adaptation_goals: list[PredictorAdaptationGoal],
            legacy_mode: bool = False,
    ):
        self.base_station_id = base_station_id
        self.model_manager = model_manager
        self.portfolio_adaptation_goals = cluster_adaptation_goals
        self.sensor_adaptation_goals = sensor_adaptation_goals
        self.initial_df = initial_df
        self.learning_strategy = learning_strategy
        self.data_reduction_strategy = data_reduction_strategy
        self._active_nodes: dict[str, NodeManager] = {}
        self._inactive_nodes: dict[str, NodeManager] = {}
        self.legacy_mode = legacy_mode

    def add_node(self,
                 node_id: str,
                 input_features: list[str],
                 output_features: list[str],
                 node_initial_df: pd.DataFrame,
                 data_reduction_strategy: DataReductionStrategy,
                 ) -> SensorKnowledgeInitialization:
        base_model = self.model_manager.base_model
        if node_initial_df is None:
            node_initial_df = self.initial_df[input_features]

        last_dt = node_initial_df.index.max()

        if node_id not in self._active_nodes:
            if node_id not in self._inactive_nodes:
                node_manager = (
                    NodeManager(node_id, input_features, output_features, base_model, data_reduction_strategy)
                )
                node_manager.last_adaptation_dt = last_dt
                node_manager.next_sync_dt = min(
                    goal.get_next_update_dt(node_manager.last_adaptation_dt, node_manager) for goal in
                    self.portfolio_adaptation_goals
                )
                for model_id in self.model_manager.get_all_model_ids():
                    node_manager.add_model(model_id)
                node_manager.set_active_model(base_model, last_dt)
                node_manager.predictor.add_measurement_df(node_initial_df)
                for goal in self.sensor_adaptation_goals:
                    node_manager.add_adaptation_goal(goal)
            else:
                node_manager = self._inactive_nodes[node_id]
            self._activate_new_node(node_manager)
            return SensorKnowledgeInitialization(
                node_id,
                base_model.metadata,
                node_manager.get_adaptation_goals(),
                node_manager.get_model_ids(),
                node_manager.next_sync_dt
            )

    def remove_node(self, node_id: str) -> None:
        if node_id in self._active_nodes:
            node_manager = self.get_node(node_id)
            node_manager.enabled = False
            self._active_nodes.pop(node_manager.node_id)
            self._inactive_nodes[node_manager.node_id] = node_manager

    def get_node(self, node_id: str) -> NodeManager:
        return self._active_nodes[node_id]

    def get_nodes(self) -> dict[str, NodeManager]:
        return self._active_nodes

    def _activate_new_node(self, node_manager: NodeManager) -> None:
        node_manager.enabled = True
        self._active_nodes[node_manager.node_id] = node_manager
