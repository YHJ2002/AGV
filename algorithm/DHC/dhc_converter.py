import numpy as np
from typing import Dict, Tuple
from collections import deque
from . import configs
from core.gridmap import GridMap
from core.agvmanager import AGVManager


class DHCCompatibleConverter:
    """
    把仓储 AGV 环境中的状态实时转换成 DHC / PRIMAL2 风格的局部观测。
    输出格式与 DHC 环境中的 `observe()` 保持兼容。
    """

    def __init__(self, num_agvs: int, gridmap: GridMap, agvmanager: AGVManager):
        self.obs_radius = configs.obs_radius
        self.padding = self.obs_radius
        self.N = num_agvs
        self.gridmap = gridmap
        self.agvmanager = agvmanager

    def convert(
        self,
        static_grid: np.ndarray,
        agv_positions_xy: Dict[int, Tuple[int, int]],
        targets: Dict[int, Tuple[Tuple[int, int], Tuple[int, int]]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回与 DHC 环境 `observe()` 一致的结果。

        Returns:
            obs: (N, 6, 2*r+1, 2*r+1) 的布尔张量
            pos: (N, 2) 的整型坐标，使用 DHC 的行列坐标系
        """
        # 外部环境一般使用 (x, y)，这里统一转成 DHC 的 (row, col)。
        agv_positions = {agv_id: (y, x) for agv_id, (x, y) in agv_positions_xy.items()}
        height, width = static_grid.shape
        active_ids = list(targets.keys())

        # 当前没有需要重规划的 AGV 时，返回空观测。
        if not active_ids:
            return (
                np.zeros((0, 6, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1), dtype=bool),
                np.zeros((0, 2), dtype=int)
            )

        # 1. 构建全局 agent 占据图。
        # 这里包含全部 AGV，因为即使某个 AGV 当前不参与规划，也依然会成为其他 AGV 的障碍。
        global_agent_map = np.zeros((height, width), dtype=bool)
        for agv_id, (cx, cy) in agv_positions.items():
            agv = self.agvmanager.get_agv(agv_id)
            size = agv.size

            for dx in range(size):
                for dy in range(size):
                    x = cx + dx
                    y = cy + dy
                    if 0 <= x < height and 0 <= y < width:
                        global_agent_map[x, y] = True

        # 2. 为每个 active AGV 构建个性化障碍图。
        personalized_obstacle_maps = np.zeros((self.N, height, width), dtype=bool)
        goal_positions = np.zeros((self.N, 2), dtype=int)

        for agv_id in active_ids:
            _, goal_pos = targets[agv_id]
            gy, gx = goal_pos
            goal_positions[agv_id] = [gx, gy]

            # 基础障碍包含墙体和货架。
            obs = (static_grid == -2) | (static_grid >= 0)

            # 当前 agent 自己的目标货架位置需要“挖空”，否则它永远无法走进去。
            if static_grid[gx, gy] >= 0:
                obs[gx, gy] = False

            personalized_obstacle_maps[agv_id] = obs

        # 3. 为每个 AGV 计算四方向 heuristic map。
        heuri_maps = self._compute_heuristic_maps(
            personalized_obstacle_maps, goal_positions, height, width, active_ids
        )

        # 4. 从全局图中裁出每个 AGV 的局部观测窗口。
        obs = np.zeros((self.N, 6, 2 * self.obs_radius + 1, 2 * self.obs_radius + 1), dtype=bool)
        padded_agent_map = np.pad(global_agent_map, self.padding, constant_values=False)
        padded_obs_maps = np.pad(
            personalized_obstacle_maps,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            constant_values=True
        )
        padded_heuri = np.pad(
            heuri_maps,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            constant_values=False
        )

        positions = np.zeros((self.N, 2), dtype=int)

        for agv_id in active_ids:
            cx, cy = agv_positions[agv_id]
            positions[agv_id] = [cx, cy]

            x1 = cx
            x2 = cx + 2 * self.obs_radius + 1
            y1 = cy
            y2 = cy + 2 * self.obs_radius + 1

            # channel 0: 其他 AGV 的占据情况。
            agent_slice = padded_agent_map[x1:x2, y1:y2].copy()
            agv = self.agvmanager.get_agv(agv_id)
            size = agv.size
            center = self.obs_radius

            # 把“自己”的 footprint 从其他 agent 图里清掉。
            for dx in range(size):
                for dy in range(size):
                    lx = center + dx
                    ly = center + dy
                    if 0 <= lx < agent_slice.shape[0] and 0 <= ly < agent_slice.shape[1]:
                        agent_slice[lx, ly] = False

            obs[agv_id, 0] = agent_slice

            # channel 1: 个性化障碍图，并加入底层地图里不可通行方向的信息。
            obstacle_slice = padded_obs_maps[agv_id, x1:x2, y1:y2]
            obstacle_slice = self._inject_unwalkable_as_obstacle(
                agv_id=agv_id,
                cx=cx,
                cy=cy,
                obstacle_slice=obstacle_slice
            )
            obs[agv_id, 1] = obstacle_slice

            # channel 2~5: 上、下、左、右四个方向的启发式图。
            obs[agv_id, 2:6] = padded_heuri[agv_id, :, x1:x2, y1:y2]

        return obs, positions

    def _compute_heuristic_maps(
        self,
        obstacle_maps: np.ndarray,
        goal_positions: np.ndarray,
        height: int,
        width: int,
        active_ids: list[int]
    ) -> np.ndarray:
        """
        计算与 DHC 观测一致的四方向 heuristic。
        如果从某格往某方向走一步后最短距离能减少 1，则该方向记为 True。
        """
        dist_maps = np.full((self.N, height, width), 2147483647, dtype=np.int32)
        heuri = np.zeros((self.N, 4, height, width), dtype=bool)

        for i in active_ids:
            gx, gy = goal_positions[i]
            if obstacle_maps[i, gx, gy]:
                # 理论上不会出现，因为目标位点已经为当前 agent 挖空。
                continue

            dist_maps[i, gx, gy] = 0
            queue = deque([(gx, gy)])

            # 从目标点反向做 BFS，得到每个可达点到目标的最短距离。
            while queue:
                x, y = queue.popleft()
                d = dist_maps[i, x, y]

                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx = x + dx
                    ny = y + dy
                    if 0 <= nx < height and 0 <= ny < width and not obstacle_maps[i, nx, ny]:
                        if dist_maps[i, nx, ny] > d + 1:
                            dist_maps[i, nx, ny] = d + 1
                            queue.append((nx, ny))

            # 根据最短距离图，为四个方向分别打标签。
            for x in range(height):
                for y in range(width):
                    if obstacle_maps[i, x, y]:
                        continue
                    d = dist_maps[i, x, y]
                    if d == 2147483647:
                        continue

                    if x > 0 and not obstacle_maps[i, x - 1, y] and dist_maps[i, x - 1, y] == d - 1:
                        heuri[i, 0, x, y] = True
                    if x < height - 1 and not obstacle_maps[i, x + 1, y] and dist_maps[i, x + 1, y] == d - 1:
                        heuri[i, 1, x, y] = True
                    if y > 0 and not obstacle_maps[i, x, y - 1] and dist_maps[i, x, y - 1] == d - 1:
                        heuri[i, 2, x, y] = True
                    if y < width - 1 and not obstacle_maps[i, x, y + 1] and dist_maps[i, x, y + 1] == d - 1:
                        heuri[i, 3, x, y] = True

        return heuri

    def _inject_unwalkable_as_obstacle(
        self,
        agv_id: int,
        cx: int,
        cy: int,
        obstacle_slice: np.ndarray
    ) -> np.ndarray:
        """
        如果底层 GridMap 判断某个方向不可通行，
        就把局部观测中该方向前方的格子额外标成障碍。
        """
        patched = obstacle_slice.copy()

        # AGV 尺寸和是否载货都会影响可通行性判断。
        agv = self.agvmanager.get_agv(agv_id)
        agv_size = agv.size
        carrying = agv.carried_box_id is not None

        # DHC 内部坐标是 (row, col)，底层 GridMap 使用 (x, y)。
        cur_x, cur_y = cy, cx
        center = self.obs_radius

        # 顺序与 heuristic channel 保持一致：上、下、左、右。
        directions = [
            (0, -1),
            (0, 1),
            (-1, 0),
            (1, 0),
        ]

        for dx, dy in directions:
            next_x = cur_x + dx
            next_y = cur_y + dy

            can_walk = self.gridmap.is_walkable(
                agv_size=agv_size,
                from_pos=(cur_x, cur_y),
                to_pos=(next_x, next_y),
                carrying_goods=carrying
            )

            if not can_walk:
                # 对应到局部观测窗口里的相对位置。
                local_x = center + dy
                local_y = center + dx

                if 0 <= local_x < patched.shape[0] and 0 <= local_y < patched.shape[1]:
                    patched[local_x, local_y] = True

        return patched
