import numpy as np
from typing import Dict, Any, Tuple
from core.gridmap import GridMap
from core.agvmanager import AGVManager
from core.ordermanager import OrderManager
from utils.logger import global_logger
from utils.simulation_clock import clock


def to_real_position(pos: Tuple[int, int], size: int = 1) -> Tuple[float, float]:
    """
    将栅格左上角坐标转换为前端显示用的真实中心坐标。
    对于尺寸大于 1 的对象，需要加入额外偏移量。
    """
    x, y = pos
    offset = (size - 1) / 2
    return (x + 0.5 + offset, y + 0.5 + offset)


def agv_to_real_center(real_pos: Tuple[float, float], size: int) -> Tuple[float, float]:
    """
    将 AGV 的左上角真实坐标转换为几何中心坐标。
    size=1 时不需要额外偏移；
    size>1 时需要根据尺寸补偿中心位置。
    """
    if size <= 1:
        return real_pos
    offset = (size - 1) / 2.0
    return (real_pos[0] + offset, real_pos[1] + offset)


def generate_send_data(
    map: GridMap,
    agvmanager: AGVManager,
    ordermanager: OrderManager,
    data_type: str = "init"
) -> Dict[str, Any]:
    """
    生成发送给前端的数据包。
    所有位置坐标统一转换为前端显示使用的真实中心坐标。
    
    data_type:
    - "init"   : 初始化数据，包含地图、AGV、货箱、接收点、等待区等静态信息
    - "update" : 运行时更新数据，包含 AGV 位置、货箱状态、路径、性能指标等动态信息
    """
    data = {}

    # =========================
    # 初始化数据：首次加载界面时发送
    # =========================
    if data_type == "init":
        data['type'] = 'init'

        # 地图尺寸信息
        data['map_size'] = {
            "width": map.width,
            "height": map.height
        }

        # AGV 初始信息：位置 + 尺寸
        agvs_info = {}
        for agv in agvmanager.all_agvs():
            agv_id = agv.id
            real_pos = agv.real_pos
            size = agv.size
            center_pos = agv_to_real_center(real_pos, size)
            agvs_info[agv_id] = {
                "pos": center_pos,
                "size": size
            }

        data['agvs'] = agvs_info

        # 地图上的货箱信息
        data['boxes'] = {
            bid: {
                "pos": to_real_position(pos, map.box_sizes.get(bid, 1)),
                "size": map.box_sizes.get(bid, 1)
            }
            for bid, pos in map.box_positions.items()
        }

        # 接收区信息
        data['receivers'] = {
            rid: {
                "pos": to_real_position(pos, map.receiver_zones_size.get(rid, 1)),
                "size": map.receiver_zones_size.get(rid, 1)
            }
            for rid, pos in map.receiver_zones.items()
        }

        # 等待区信息
        data['wait_zones'] = {
            wid: {
                "pos": to_real_position(pos, map.wait_zones_size.get(wid, 1)),
                "size": map.wait_zones_size.get(wid, 1)
            }
            for wid, pos in map.wait_zones.items()
        }

        # 障碍物信息
        data['obstacles'] = [to_real_position(pos) for pos in map.obstacles]

        # 初始化订单面板数据
        data['orders'] = {
            "counts": {
                "unprocessed": len(ordermanager.unprocessed_orders),
                "processing": len(ordermanager.processing_orders),
                "completed": len(ordermanager.finished_orders),
            },
            # 初始化时日志面板为空
            "logs": {"generation": [], "assignment": [], "completion": []},
            # 每个 AGV 当前订单执行进度
            "agv_progress": global_logger.get_agv_order_progress(agvmanager),
        }

    # =========================
    # 更新数据：仿真运行过程中持续发送
    # =========================
    elif data_type == "update":
        data['type'] = 'update'

        # 当前所有 AGV 的中心位置
        agv_pos = {
            aid: agv_to_real_center(pos, agvmanager.get_agv_size(aid))
            for aid, pos in agvmanager.get_all_real_positions().items()
        }
        data['agvs'] = agv_pos

        # 当前各 AGV 携带的货箱编号
        carrying_status = agvmanager.get_carried_box_ids()

        # 已经被 AGV 搬运中的货箱位置
        boxes_on_agv = {}

        # 仍然在货架上的货箱位置
        boxes_on_shelf = {}

        # 将正在被 AGV 搬运的货箱绑定到 AGV 当前坐标
        for agv_id, b_id in carrying_status.items():
            if b_id is not None:
                boxes_on_agv[b_id] = agv_pos[agv_id]

        # 找出仍在货架上的货箱
        boxes_on_shelf_id_set = map.box_id_set - set(boxes_on_agv.keys())
        for b_id in boxes_on_shelf_id_set:
            shelf_pos = map.box_positions[b_id]
            real_pos = to_real_position(shelf_pos, map.box_sizes.get(b_id, 1))
            boxes_on_shelf[b_id] = real_pos

        data['boxes_on_agv'] = boxes_on_agv
        data['boxes_on_shelf'] = boxes_on_shelf

        # 当前安全路径/动态占用区域，用于前端显示路径或安全区域
        safe_paths = {}
        for agv_id, occupied_set in map.dynamic_occupied.items():
            safe_paths[agv_id] = [to_real_position(pos) for pos in occupied_set]
        data['safe_paths'] = safe_paths

        # 运行时性能指标，如吞吐量、冲突率等
        data['metrics'] = global_logger.get_runtime_metrics(clock.now())

        # 订单面板更新数据
        data['orders'] = {
            "counts": {
                "unprocessed": len(ordermanager.unprocessed_orders),
                "processing": len(ordermanager.processing_orders),
                "completed": len(ordermanager.finished_orders),
            },
            # 获取订单日志，用于前端面板展示
            "logs": global_logger.get_order_logs_for_panel(),
            # 获取 AGV 当前订单执行进度
            "agv_progress": global_logger.get_agv_order_progress(agvmanager),
        }

    return data