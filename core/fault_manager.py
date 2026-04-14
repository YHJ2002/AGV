# core/fault_manager.py
from typing import Optional, Dict, Tuple, List
import random
from core.agvmanager import AGVManager
from core.env import Env
from core.gridmap import GridMap
from config.settings import FaultConfig
from utils.logger import global_logger
from utils.simulation_clock import clock


class FaultManager:
    def __init__(self, agv_manager: AGVManager, env: Env, gridmap: GridMap):
        # 保存依赖对象，故障管理需要访问 AGV、环境和地图
        self.agv_manager = agv_manager
        self.gridmap = gridmap
        self.env = env

        # 从环境中读取静态栅格地图，用于后续故障撤离路径规划
        env_info = env.get_env_info()
        self.static_grid = env_info['static_grid']

        # 初始化故障配置和运行时状态
        self.reset()

    def step(self):
        """
        每个仿真步调用一次。
        负责：
        1. 按概率注入新的随机故障
        2. 更新已故障 AGV 的维修倒计时
        """
        if not self.enable_faults:
            return

        self._maybe_trigger_fault()
        self._update_repairs()

    def _maybe_trigger_fault(self):
        """
        按配置概率触发随机故障。
        若不允许多故障同时存在，则当前已有故障时不再触发新故障。
        """
        if not self.allow_multiple_faults and self.active_faults:
            return

        for agv_id in self.agv_manager.all_agv_ids:
            # 已经处于故障中的 AGV 跳过
            if agv_id in self.active_faults:
                continue

            # 按 fault_prob 概率触发故障
            if self.rng.random() < self.fault_prob:
                self._trigger_fault(agv_id)
                if not self.allow_multiple_faults:
                    break

    def _trigger_fault(self, agv_id: int):
        """
        为指定 AGV 生成一次故障，并随机分配维修时间。
        """
        repair_time = self.rng.randint(
            self.mean_repair_time // 2,
            self.mean_repair_time * 2,
        )

        # 记录该 AGV 的剩余维修时间
        self.active_faults[agv_id] = repair_time

        # 执行故障模拟逻辑
        self.simulate_fault(agv_id)

    def _update_repairs(self):
        """
        更新所有故障 AGV 的维修倒计时。
        当剩余维修时间降到 0 时，触发修复。
        """
        recovered = []

        for agv_id in self.active_faults:
            self.active_faults[agv_id] -= 1
            if self.active_faults[agv_id] <= 0:
                recovered.append(agv_id)

        for agv_id in recovered:
            self.repair_agv(agv_id)
            del self.active_faults[agv_id]

    def handle_message(self, msg: dict):
        """
        处理前端传来的故障控制命令。
        例如：
        {"cmd": "damage", "agv_id": 2}
        {"cmd": "repair", "agv_id": 2}
        """
        cmd = msg.get("cmd")
        agv_id = msg.get("agv_id")
        print(f"[FaultManager] Handling command: {msg}")

        if cmd == "damage":
            self.simulate_fault(agv_id)
        elif cmd == "repair":
            self.repair_agv(agv_id)

        print("Command processed.")

    def simulate_fault(self, agv_id: int):
        """
        模拟 AGV 故障。
        处理逻辑：
        1. 将 AGV 设置为不可工作
        2. 找到最近的边界可通行栅格作为撤离目标
        3. 规划一条维修路径
        4. 将该路径写入地图的动态占用信息中
        """
        self.agv_manager.set_agv_status(agv_id, False)

        agv_grid_pos = self.agv_manager.get_agv(agv_id).grid_pos

        # 寻找最近的边界空闲点
        border_cell = self._find_nearest_border_free_cell(agv_grid_pos)

        # 规划故障撤离路径
        path = self.plan_repair_path(agv_grid_pos, border_cell)

        # 将路径加入动态占用，表示该故障 AGV 占据这条维修/撤离通道
        self.gridmap.add_dynamic_occupancy(str(agv_id), path)

        global_logger.add_runtime_log(
            f"[FaultManager] AGV {agv_id} failed at step {clock.now()}"
        )

    def repair_agv(self, agv_id: int):
        """
        修复 AGV。
        处理逻辑：
        1. 恢复 AGV 工作状态
        2. 从地图中移除故障 AGV 的动态占用路径
        """
        self.agv_manager.set_agv_status(agv_id, True)
        self.gridmap.remove_dynamic_occupancy(str(agv_id))

        global_logger.add_runtime_log(
            f"[FaultManager] AGV {agv_id} repaired at step {clock.now()}"
        )

    def assign_replacement(self, faulty_agv_id: int, replacement_agv_id: int):
        """
        为故障 AGV 分配替代 AGV。
        当前为预留接口，尚未实现。
        后续可扩展为任务重分配逻辑。
        """
        pass

    def _find_nearest_border_free_cell(self, start: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        从起点出发，搜索最近的边界可通行栅格。
        约束：
        - 只能在 static_grid 中值为 -1 的可通行区域搜索
        - 不能穿过 receiver_zones（接收区）
        """
        h, w = self.static_grid.shape
        visited = set()
        queue = [start]
        receiver_set = set(self.gridmap.receiver_zones.values())

        while queue:
            x, y = queue.pop(0)

            if (x, y) in visited:
                continue
            visited.add((x, y))

            # 若当前点是边界可通行点，且不是接收区，则返回
            if self.static_grid[y, x] == -1 and (x == 0 or y == 0 or x == w - 1 or y == h - 1):
                if (x, y) not in receiver_set:
                    return (x, y)

            # 四邻域 BFS 扩展
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < w and 0 <= ny < h and
                    self.static_grid[ny, nx] == -1 and
                    (nx, ny) not in visited and
                    (nx, ny) not in receiver_set
                ):
                    queue.append((nx, ny))

        return None

    def plan_repair_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        在 static_grid 上为故障 AGV 规划一条从当前点到边界点的撤离路径。
        使用 BFS 搜索，避免穿过接收区。
        """
        h, w = self.static_grid.shape
        queue = [(start, [start])]
        visited = set()
        receiver_set = set(self.gridmap.receiver_zones.values())

        while queue:
            (x, y), path = queue.pop(0)

            if (x, y) == goal:
                return path

            if (x, y) in visited:
                continue
            visited.add((x, y))

            # 四邻域 BFS 扩展
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < w and 0 <= ny < h and
                    self.static_grid[ny, nx] == -1 and
                    (nx, ny) not in visited and
                    (nx, ny) not in receiver_set
                ):
                    queue.append(((nx, ny), path + [(nx, ny)]))

        return None

    def reset(self):
        """
        重置故障管理器状态。
        主要包括：
        1. 重置随机数种子
        2. 从 FaultConfig 同步故障配置
        3. 清空当前活动故障列表
        """
        self.rng = random.Random(FaultConfig.fault_seed)

        # 从 FaultConfig 同步配置
        self.enable_faults = FaultConfig.enable_faults
        self.fault_prob = FaultConfig.fault_prob
        self.mean_repair_time = FaultConfig.mean_repair_time
        self.allow_multiple_faults = FaultConfig.allow_multiple_faults

        # 运行时状态：记录当前所有处于故障中的 AGV 及其剩余维修时间
        self.active_faults: Dict[int, int] = {}