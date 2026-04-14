from collections import deque
from typing import Deque, List, Tuple, Optional, Dict, Set, Generator
from enum import Enum, auto
from core.gridmap import GridMap  # 栅格地图实例，提供空间环境与货物信息
from core.ordermanager import OrderManager  # 订单管理器，处理订单生命周期
from config.settings import SimConfig  # 全局仿真配置（AGV速度、转向时间等）
import json

epsilon = 1e-4  # 浮点数精度阈值，用于位置对齐判断

class Direction(Enum):
    """AGV运动方向枚举（与栅格坐标系匹配）"""
    UP = auto()    # 上（y轴负向）
    DOWN = auto()  # 下（y轴正向）
    LEFT = auto()  # 左（x轴负向）
    RIGHT = auto() # 右（x轴正向）

class AGVAction(Enum):
    """AGV作业动作枚举（对应仓储核心操作）"""
    PICK = "pick"        # 取货：从当前栅格获取货物
    PLACE = "place"      # 放货：将货物放置到当前栅格
    HANDOVER = "handover"# 交接：向订单系统完成货物交付

class StepInfo(Enum):
    """AGV单步状态枚举（用于仿真日志与前端可视化）"""
    MOVE = auto()            # 移动中
    COLLISION = auto()       # 碰撞（预留扩展）
    STAY_OFF_GOAL = auto()   # 非目标位置静止
    STAY_ON_GOAL = auto()    # 目标位置静止
    FINISH = auto()          # 任务节点完成
    TURNING = auto()         # 转向中
    OTHER = auto()           # 其他状态

class AGV:
    """AGV核心类：封装属性管理、运动控制、任务执行逻辑"""
    def __init__(
        self,
        agv_id: int,               # AGV唯一ID
        size:int,                  # AGV尺寸（需与货物尺寸匹配）
        init_grid_pos: Tuple[int, int],  # 初始栅格位置 (x, y)
        map_inst: GridMap,         # 栅格地图实例
        order_manager: OrderManager,      # 订单管理器实例
    ):
        self.id: int = agv_id
        self.size: int = size
        self.init_grid_pos: Tuple[int, int] = init_grid_pos  # 初始位置备份
        self.grid_pos: Tuple[int, int] = init_grid_pos  # 当前栅格位置
        self.real_pos: Tuple[float, float] = (init_grid_pos[0] + 0.5, init_grid_pos[1] + 0.5)  # 物理坐标（栅格中心为基准）
        self.map = map_inst
        self.order_manager = order_manager
        
        # 任务队列：(目标栅格, 动作类型, 额外参数)，双端队列保证执行顺序
        self.task_queue: Deque[Tuple[Tuple[int, int], AGVAction, Optional[int]]] = deque()
        self.rest_target: Optional[Tuple[int, int]] = None  # 休息区位置（无任务时前往）
        self.action_queue: Deque[Tuple[int, int]] = deque()  # 路径动作队列（待移动的栅格序列）
        
        # 运动相关参数（从全局配置加载）
        self.speed: float = 0.0  # 当前速度
        self.max_speed: float = SimConfig.agv_max_speed  # 最大移动速度
        self.time_step: float = SimConfig.time_step  # 仿真单步时间
        self.direction: Optional[Direction] = Direction.LEFT  # 当前运动方向
        turning_time_90: float = SimConfig.agv_turn_time_90  # 90度转向耗时
        self.turning_steps_90: int = int(round(turning_time_90 / self.time_step))  # 转向所需步数
        self.turning_timer: int = 0  # 转向计时器（倒计时）
        self.target_direction: Optional[Direction] = None  # 目标运动方向

        self.carried_box_id: int = None  # 携带货物ID（无货物时为None）
        self.is_working: bool = True  # 工作状态标记（True为正常工作）
        self.last_completed_task_pos: Tuple[int, int] = init_grid_pos  # 上一任务完成位置（用于进度计算）

    @property
    def is_idle(self) -> bool:
        """是否空闲：无待执行任务"""
        return len(self.task_queue) == 0

    @property
    def is_resting(self) -> bool:
        """是否在休息区：已设置休息区且当前位置匹配"""
        return self.rest_target is not None and self.grid_pos == self.rest_target
    
    @property
    def is_aligned(self) -> bool:
        """物理坐标是否与栅格中心对齐（避免浮点误差）"""
        gx, gy = self.grid_pos
        expected_x, expected_y = gx + 0.5, gy + 0.5
        rx, ry = self.real_pos
        return abs(rx - expected_x) < epsilon and abs(ry - expected_y) < epsilon

    def step(self, next_grid: Tuple[int, int]) -> Tuple[bool, StepInfo]:
        """单步执行核心逻辑：更新位置、执行任务，返回是否需要重规划与当前状态"""
        if not self.is_working:
            return False, StepInfo.OTHER
        
        # 优先处理转向（转向中不移动）
        if self.turning_timer > 0:
            self.turning_timer -= 1
            if self.turning_timer == 0:
                self.direction = self.target_direction  # 转向完成，更新当前方向
            return False, StepInfo.TURNING
               
        # 更新物理位置
        _, step_info = self.update_position(next_grid)
        replan_required = False  # 是否需要重新规划路径

        # 路径动作队列推进：到达下一个路径节点，移除该节点
        if self.action_queue and self.grid_pos == self.action_queue[0]:
            self.action_queue.popleft()
            # 任务队列执行：到达任务节点，执行对应动作
            if self.task_queue and self.grid_pos == self.task_queue[0][0]:
                task_pos, action, extra = self.task_queue.popleft()
                self.last_completed_task_pos = task_pos
                self._execute_action(action, extra)  # 执行取货/放货/交接动作
                replan_required = True  # 任务完成，需要重规划下一任务路径
                step_info = StepInfo.FINISH
        # 无路径队列但有任务：直接执行当前任务
        elif not self.action_queue and self.task_queue and self.grid_pos == self.task_queue[0][0]:
            task_pos, action, extra = self.task_queue.popleft()
            self.last_completed_task_pos = task_pos
            self._execute_action(action, extra)
            replan_required = True
            step_info = StepInfo.FINISH
        # 无路径队列且非休息状态：需要重规划
        if not self.action_queue and not self.is_resting:
            replan_required = True

        return replan_required, step_info

    def update_position(self, next_grid: Tuple[int, int]) -> tuple[bool, StepInfo]:
        """更新物理位置：计算方向、转向、移动偏移，返回是否移动及状态"""
        dx = next_grid[0] - self.grid_pos[0]  # x方向栅格差
        dy = next_grid[1] - self.grid_pos[1]  # y方向栅格差

        step_info = StepInfo.MOVE
        # 根据栅格差确定目标方向
        if dx == 1:
            self.target_direction = Direction.RIGHT
        elif dx == -1:
            self.target_direction = Direction.LEFT
        elif dy == 1:
            self.target_direction = Direction.DOWN
        elif dy == -1:
            self.target_direction = Direction.UP
        else:
            self.target_direction = None
            step_info = StepInfo.STAY_OFF_GOAL  # 无栅格差，静止状态

        # 计算转向步数，若需要转向则直接返回
        self.turning_timer = self._calculate_turn_time(self.direction, self.target_direction)
        if self.turning_timer > 0:
            self.turning_timer -= 1
            if self.turning_timer == 0:
                self.direction = self.target_direction
            return False, StepInfo.TURNING
        
        # 计算移动偏移量（速度×时间步）
        self.speed = self.max_speed
        offset = self.speed * self.time_step
        x, y = self.real_pos
        moved = False  # 是否完成栅格移动

        # X方向移动（横向）
        if dx != 0:
            target_x = next_grid[0] + 0.5  # 目标栅格中心x坐标
            if abs(target_x - x) <= offset + epsilon:  # 一步可到达，直接定位到中心
                x = target_x
                moved = True
            else:  # 分步移动，更新偏移
                x += offset * dx
        # Y方向移动（纵向）
        if dy != 0:
            target_y = next_grid[1] + 0.5  # 目标栅格中心y坐标
            if abs(target_y - y) <= offset + epsilon:
                y = target_y
                moved = True
            else:
                y += offset * dy

        # 更新物理位置，若移动完成则更新栅格位置
        self.real_pos = (x, y)
        if moved:
            self.grid_pos = next_grid
        return moved, step_info

    def _pick_box(self):
        """取货动作：从当前栅格获取货物，校验尺寸匹配"""
        box_id = self.map.pick_box_at(self.grid_pos)  # 从地图获取当前栅格货物ID
        box_size = self.map.box_sizes.get(box_id, 1)
        # 尺寸不匹配则抛出异常
        if box_id is not None and box_size != self.size:
            raise ValueError(f"AGV {self.id} 尺寸不匹配货物 {box_id}（AGV={self.size}, 货物={box_size}）")
        if box_id is not None:
            self.carried_box_id = int(box_id)  # 记录携带货物ID

    def _place_box(self):
        """放货动作：将携带货物放置到当前栅格"""
        if self.carried_box_id is not None:
            # 调用地图接口放置货物，成功则清空携带记录
            success = self.map.place_box_at(self.grid_pos, self.carried_box_id)
            if success:
                self.carried_box_id = None

    def _handover_box(self, order_id: Optional[int]):
        """交接动作：向订单管理器上报订单完成"""
        if self.carried_box_id is None or order_id is None:
            return
        # 调用订单管理器接口，完成订单闭环
        self.order_manager.complete_order(
            order_id=order_id, agv_id=self.id, box_id=self.carried_box_id, agv_pos=self.grid_pos
        )
        self.carried_box_id = None  # 交接后清空携带记录

    def assign_task(self, task_positions: List[Tuple[Tuple[int, int], AGVAction, Optional[int]]]):
        """分配任务：更新任务队列，取消休息状态"""
        self.task_queue = deque(task_positions)
        self.rest_target = None

    def assign_rest_zone(self, rest_pos: Tuple[int, int]):
        """分配休息区：设置休息目标位置"""
        self.rest_target = rest_pos

    def set_new_plan(self, path: List[Tuple[int, int]]):
        """设置路径规划结果：更新动作队列（栅格序列）"""
        self.action_queue = deque(path)

    def get_next_pos(self) -> Tuple[int, int]:
        """获取下一移动栅格：无路径则返回当前位置"""
        return self.action_queue[0] if self.action_queue else self.grid_pos

    def _execute_action(self, action: AGVAction, extra: Optional[int]):
        """动作执行分发：根据动作类型调用对应私有方法"""
        if action == AGVAction.PICK:
            self._pick_box()
        elif action == AGVAction.PLACE:
            self._place_box()
        elif action == AGVAction.HANDOVER:
            self._handover_box(extra)

    def reset(self):
        """重置AGV到初始状态（用于仿真重启）"""
        self.grid_pos = self.init_grid_pos
        self.real_pos = (self.init_grid_pos[0] + 0.5, self.init_grid_pos[1] + 0.5)
        self.task_queue.clear()
        self.action_queue.clear()
        self.rest_target = None
        self.carried_box_id = None
        self.is_working = True
        self.direction = None
        self.last_completed_task_pos = self.init_grid_pos
    
    def _calculate_turn_time(self, current_direction: Direction, target_direction: Optional[Direction],) -> int:
        """计算转向所需步数：同向0步，垂直90度，反向180度"""
        if target_direction is None or current_direction == target_direction:
            return 0
        # 反向方向映射表
        opposites = {
            Direction.UP: Direction.DOWN,
            Direction.DOWN: Direction.UP,
            Direction.LEFT: Direction.RIGHT,
            Direction.RIGHT: Direction.LEFT
        }
        if opposites.get(current_direction) == target_direction:
            return self.turning_steps_90 * 2  # 反向（180度）需2倍90度转向步数
        return self.turning_steps_90  # 垂直（90度）需1倍步数