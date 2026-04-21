# 导入工具：数据类、类型注解、JSON、枚举
from dataclasses import dataclass
from typing import Optional
import json
from enum import Enum

# ========================= 枚举定义：只能选固定值 =========================
# 枚举 = 给选项起名字，防止写错字符串

class SchedulerType(Enum):
    """调度算法类型：AGV 任务分配用哪种算法"""
    RANDOM = "random"   # 随机分配（最简单，做对比基准）
    TA = "ta"           # 任务分配算法（你当前用的核心算法）

class PlannerType(Enum):
    """路径规划算法类型：AGV 怎么走用哪种算法"""
    ASTAR = "astar"     # A* 算法（基础、快）
    CBS_FW = "cbs_fw"   # 冲突搜索算法（避冲突）
    DHC = "dhc"         # 深度学习路径规划

class OrderMode(Enum):
    """所有支持的订单生成模式"""
    ONESHOT = "oneshot"                          # 一次性生成所有订单（默认、最简单）
    CONTINUOUS_CONSTANT = "continuous_constant"  # 持续匀速生成
    CONTINUOUS_PERIODIC = "continuous_periodic"  # 周期性波浪生成（高峰+低峰）
    CONTINUOUS_PARETO = "continuous_pareto"      # 帕累托分布（热点货架）
    CONTINUOUS_BURST = "continuous_burst"        # 随机爆单模式（促销）

# ========================= 主仿真配置 =========================
@dataclass
class SimConfig:
    """仿真总配置：所有核心参数都在这里"""

    # ==================== 算法选择 ====================
    scheduler_type: SchedulerType = SchedulerType.TA
    # 调度算法：当前用 TA 算法

    planner_type: PlannerType = PlannerType.CBS_FW
    # 路径规划：当前用 A*
    # 可切换：ASTAR / CBS_FW / DHC

    force_replan_every_step: bool = False
    # 是否每一步都强制重新规划路径
    # 开启 = 更灵活，但计算量大

    dhc_model_path: str = '.\\algorithm\\DHC\\models\\36000.pth'
    # DHC 深度学习模型路径

    # ==================== 订单参数 ====================
    order_mode: OrderMode = OrderMode.ONESHOT
    # 订单生成模式：当前是一次性生成

    total_orders_limit = 50
    # 订单总数上限：先设 50 跑通，调试完再加大

    size2_ratio: float = 0.2
    # 双货物订单比例：20% 的订单一次要搬两个货

    order_processing_timeout: int = 30
    # 订单超时时间：30 步没完成就重新排队

    order_seed: Optional[int] = 42
    # 订单随机种子：固定值 = 每次生成一样的订单

    # ==================== 地图与仿真步数 ====================
    map_file: str = "config/maps/map_25_20_het.json"
    # 地图文件路径

    max_steps: int = 1000
    # 仿真最大步数：到 1000 步自动停止

    time_step: float = 1.0
    # 每一步代表现实 1 秒

    agv_max_speed: float = 1
    # AGV 最大速度：每步走 1 格

    agv_turn_time_90: float = 0
    # AGV 90 度转向耗时：当前不耗时

    # ==================== 前端可视化界面大小 ====================
    cell_size: int = 40         # 地图每个格子 40px
    panel_width: int = 300       # 右侧控制面板宽度 300px

    # ==================== 日志 ====================
    log_to_file: bool = True             # 是否写日志文件
    log_dir: str = "logs"                # 日志文件夹
    log_file_name: str = "simulation.log"# 日志文件名
    log_overwrite: bool = True           # 重置时覆盖日志
    log_to_console: bool = False         # 不在控制台打印日志

# ========================= 故障配置 =========================
@dataclass
class FaultConfig:
    """AGV 故障配置：是否坏车、坏车概率、维修时间"""

    enable_faults: bool = False
    # 是否开启故障：当前关闭（调试更稳定）

    fault_prob: float = 0.01
    # 每步 AGV 故障概率：1%

    mean_repair_time: int = 40
    # 平均维修时间：40 步

    allow_multiple_faults: bool = False
    # 是否允许多台 AGV 同时故障

    fault_seed: Optional[int] = 42
    # 故障随机种子

# ========================= 各种订单模式的详细配置 =========================
# 不同的订单生成方式，各自有专属参数

@dataclass
class OneShotConfig:
    """一次性订单模式：仿真开始前生成所有订单"""
    # 无额外参数，最简单
    pass

@dataclass
class ContinuousConstantConfig:
    """持续匀速生成订单：每隔固定步数生成一批"""
    batch_size: int = 10                    # 每批 10 单
    generation_interval_steps: int = 50     # 每 50 步生成一批

@dataclass
class ContinuousPeriodicConfig:
    """周期性波浪订单：高峰、低峰交替"""
    base_batch_size: int = 10               # 基础每批 10 单
    generation_interval_steps: int = 20     # 每 20 步一波
    cycle_duration_steps: int = 80          # 一个周期 80 步
    peak_multiplier: float = 3.0            # 高峰期 ×3 订单
    valley_multiplier: float = 0.3          # 低峰期 ×0.3 订单
    wave_type: str = "sine"                # 波浪类型：平滑正弦

@dataclass
class ContinuousParetoConfig:
    """帕累托分布：少数热点货架产生大量订单（20/80原则）"""
    alpha: float = 2.0
    scale: float = 10.0
    generation_interval_steps: int = 30
    hot_sku_percentage: float = 0.2         # 20% 热点货架
    hot_sku_multiplier: float = 5.0         # 热点货架订单 ×5

@dataclass
class ContinuousBurstConfig:
    """随机爆单模式：模拟促销、突然大量订单"""
    base_batch_size: int = 10
    generation_interval_steps: int = 60
    burst_probability_per_1000_steps: int = 50   # 爆单概率
    burst_duration_steps: int = 1800             # 爆单持续时间
    burst_peak_batch_size: int = 50              # 爆单每批 50 单
    burst_interval_steps: int = 5
    total_orders_limit: Optional[int] = None