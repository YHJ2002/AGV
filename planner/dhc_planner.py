# planner/dhc_planner.py
import torch
import numpy as np
from typing import Dict, Tuple, List, Set
from collections import defaultdict
import os

from core.env import Env
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from core.fault_manager import FaultManager
from core.agvmanager import AGVManager
from planner.base_planner import BasePlanner
from algorithm.DHC.dhc_converter import DHCCompatibleConverter
from algorithm.DHC.model import Network
from algorithm.DHC.dhc_env import ACTION_DELTA


class DHCPlanner(BasePlanner):
    """
    DHC 学习型路径规划器。
    该规划器依赖训练好的神经网络模型输出动作，
    本身不做预约表冲突检查，冲突过滤主要依赖环境模块处理。
    """

    def __init__(
        self,
        env: Env,
        agv_manager: AGVManager,
        order_manager: OrderManager,
        map: GridMap,
        fault_manager: FaultManager,
        model_path: str,
        forward_steps: int = 6,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        # 调用父类初始化，保存环境、AGV 管理器、订单管理器、地图和故障管理器
        super().__init__(env, agv_manager, order_manager, map, fault_manager)

        # 推理设备：优先 GPU，否则 CPU
        self.device = device

        # 每次规划向前展开的步数
        self.forward_steps = forward_steps

        # DHC 输入格式转换器
        # 负责把当前系统环境状态转换成 DHC 模型需要的 observation 和 position 格式
        self.converter = DHCCompatibleConverter(
            num_agvs=self.env.agv_manager.num_agvs,
            gridmap=self.env.map,
            agvmanager=self.env.agv_manager
        )

        # 创建 DHC 神经网络模型，并切换到推理模式
        self.model = Network().to(self.device)
        self.model.eval()

        # 检查模型权重文件是否存在
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"DHC model not found: {model_path}")

        # 加载训练好的模型参数
        state_dict = torch.load(model_path, map_location=device)
        self.model.load_state_dict(state_dict)

        print(f"[DHCPlanner] Loaded weights: {model_path}")

    def plan(
        self,
        targets: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
        scheduler
    ) -> Dict[int, List[Tuple[int, int]]]:
        """
        DHC 路径规划入口。

        输入：
            targets = {agv_id: (current_pos, goal_pos)}

        输出：
            {agv_id: [next_pos1, next_pos2, ...]}
            每个 AGV 返回一个长度不超过 forward_steps 的前向路径序列。
        """
        if not targets:
            return {}

        # 从环境中读取静态地图和所有 AGV 当前网格位置
        env_info = self.env.get_env_info()
        static_grid = env_info['static_grid']
        current_positions = env_info['current_grid_pos']

        # 将当前系统状态转换为 DHC 模型需要的输入格式
        obs_dhc, pos_dhc = self.converter.convert(
            static_grid=static_grid,
            agv_positions_xy=current_positions,
            targets=targets
        )

        # 关闭梯度计算，执行模型推理
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_dhc).float().to(self.device)
            pos_tensor = torch.from_numpy(pos_dhc).long().to(self.device)

            # 模型输出动作编号
            actions, _, _, _ = self.model.step(obs_tensor, pos_tensor)

        paths: Dict[int, List[Tuple[int, int]]] = {}
        active_ids = list(targets.keys())

        # 将模型输出的动作编号转换为具体路径
        for idx, agv_id in enumerate(active_ids):
            start_x, start_y = targets[agv_id][0]

            # 根据动作编号查表得到位移增量
            dx, dy = ACTION_DELTA[actions[idx]]

            path = []
            cur_x, cur_y = start_x, start_y

            # 以同一动作连续展开 forward_steps 步
            for _ in range(self.forward_steps):
                next_x = cur_x + dx
                next_y = cur_y + dy
                path.append((next_x, next_y))
                cur_x, cur_y = next_x, next_y

            paths[agv_id] = path

        return paths