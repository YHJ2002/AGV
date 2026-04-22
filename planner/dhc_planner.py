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

        self.model = Network().to(self.device)
        self.model.eval()

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DHC model not found: {model_path}")

        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)
        self.model.reset()

        print(f"[DHCPlanner] Loaded weights: {model_path}")

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
        active_ids = list(targets.keys())

        # DHC predicts a single next action, so only emit one next cell here.
        for idx, agv_id in enumerate(active_ids):
            start_x, start_y = targets[agv_id][0]
            dx, dy = ACTION_DELTA[actions[idx]]
            paths[agv_id] = [(start_x + dx, start_y + dy)]

        return paths
