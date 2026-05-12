import os
from typing import Dict, List, Tuple

import torch

from algorithm.DHC.dhc_converter import DHCCompatibleConverter
from algorithm.DHC.dhc_env import ACTION_DELTA
from algorithm.DHC.model import Network
from core.agvmanager import AGVManager
from core.env import Env
from core.fault_manager import FaultManager
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from planner.base_planner import BasePlanner


class DHCPlanner(BasePlanner):
    @staticmethod
    def _infer_arch_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        cnn_channel = state_dict["obs_encoder.0.weight"].shape[0]
        hidden_dim = state_dict["state.weight"].shape[1]
        return cnn_channel, hidden_dim

    def __init__(
        self,
        env: Env,
        agv_manager: AGVManager,
        order_manager: OrderManager,
        map: GridMap,
        fault_manager: FaultManager,
        model_path: str,
        forward_steps: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(env, agv_manager, order_manager, map, fault_manager)
        self.device = device
        self.forward_steps = forward_steps

        self.converter = DHCCompatibleConverter(
            num_agvs=self.env.agv_manager.num_agvs,
            gridmap=self.env.map,
            agvmanager=self.env.agv_manager,
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DHC model not found: {model_path}")

        state_dict = torch.load(model_path, map_location=device)
        cnn_channel, hidden_dim = self._infer_arch_from_state_dict(state_dict)

        self.model = Network(cnn_channel=cnn_channel, hidden_dim=hidden_dim).to(self.device)
        self.model.eval()
        self.model.load_state_dict(state_dict)
        self.model.reset()

        print(
            f"[DHCPlanner] Loaded weights: {model_path} "
            f"(cnn_channel={cnn_channel}, hidden_dim={hidden_dim})"
        )

    def plan(
        self,
        targets: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
        scheduler,
    ) -> Dict[int, List[Tuple[int, int]]]:
        if not targets:
            return {}

        env_info = self.env.get_env_info()
        obs_dhc, pos_dhc = self.converter.convert(
            static_grid=env_info["static_grid"],
            agv_positions_xy=env_info["current_grid_pos"],
            targets=targets,
        )

        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_dhc).float().to(self.device)
            pos_tensor = torch.from_numpy(pos_dhc).long().to(self.device)
            actions, _, _, _ = self.model.step(obs_tensor, pos_tensor)

        paths: Dict[int, List[Tuple[int, int]]] = {}
        active_ids = sorted(targets.keys())

        # DHC predicts a single next action, so only emit one next cell here.
        for agv_id in active_ids:
            start_x, start_y = targets[agv_id][0]
            dx, dy = ACTION_DELTA[actions[agv_id]]
            paths[agv_id] = [(start_x + dx, start_y + dy)]

        return paths
