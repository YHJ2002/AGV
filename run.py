# run.py
import asyncio
import json
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

from config.settings import SimConfig
from core.agvmanager import AGVManager
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from core.env import Env
from core.simulator import Simulator
from core.data_generator import generate_send_data
from core.fault_manager import FaultManager
from utils.logger import global_logger
import websockets
from utils.simulation_clock import clock
from utils.algorithm_factory import build_scheduler, build_planner

# 全局运行状态
STATE = {
    "paused": True,        # 是否暂停
    "step_trigger": False  # 是否触发单步执行
}
RUNNING = True
NEED_RESET = False


def start_http_server(port=8000):
    """启动本地 HTTP 静态服务，避免 file:// 带来的跨域问题。"""
    os.chdir(os.path.abspath("."))
    handler = SimpleHTTPRequestHandler
    with TCPServer(("", port), handler) as httpd:
        print(f"HTTP server running at http://localhost:{port}")
        httpd.serve_forever()


async def simulator_loop(websocket, message_queue):
    """
    仿真主循环：
    1. 初始化核心对象
    2. 首次向前端发送 init 数据
    3. 进入仿真推进 / 重置 / 等待循环
    """
    global RUNNING
    global NEED_RESET

    print("Simulation begin")

    # -------------------- 1) 核心对象初始化 --------------------
    grid_map = GridMap()
    ordermanager = OrderManager(grid_map)
    agv_manager = AGVManager(grid_map, ordermanager)
    env = Env(agv_manager, grid_map, ordermanager)
    fault_manager = FaultManager(agv_manager, env, grid_map)

    scheduler = build_scheduler(env, agv_manager, ordermanager, grid_map, fault_manager)
    planner = build_planner(env, agv_manager, ordermanager, grid_map, fault_manager)

    simulator = Simulator(grid_map, agv_manager, ordermanager, env, scheduler, planner)

    # 首次发送初始化数据，前端据此创建地图、AGV、货架、接收区等对象
    init_data = generate_send_data(grid_map, agv_manager, ordermanager, data_type="init")
    await websocket.send(json.dumps(init_data))

    # -------------------- 2) 仿真生命周期循环 --------------------
    while RUNNING:
        # -------------------- reset 热重置 --------------------
        if NEED_RESET:
            print("Resetting simulation...")

            # A. 先暂停，防止重置过程中继续推进
            STATE["paused"] = True
            STATE["step_trigger"] = False

            # B. 先通知前端清空旧场景
            # 这一条很关键，否则前端如果直接接收新的 init 并追加渲染，
            # 就会出现“旧 AGV 图像 + 新 AGV 图像同时存在”的问题
            await websocket.send(json.dumps({"type": "reset"}))

            # C. 重置全局时钟和日志
            clock.reset()
            global_logger.reset()

            # D. 重置环境对象。
            # env.reset() 内部已经会重置 AGV / 地图 / 订单，避免重复重置导致状态不一致。
            env.reset()
            fault_manager.reset()        # 重置故障管理状态

            # E. 重新创建调度器和规划器
            # 避免它们内部仍持有旧状态
            scheduler = build_scheduler(env, agv_manager, ordermanager, grid_map, fault_manager)
            planner = build_planner(env, agv_manager, ordermanager, grid_map, fault_manager)

            # F. 重新创建仿真器，确保引用的是重置后的对象
            simulator = Simulator(grid_map, agv_manager, ordermanager, env, scheduler, planner)

            # G. 清空消息队列，避免 reset 前残留的 fault 命令继续生效
            while not message_queue.empty():
                _ = await message_queue.get()

            # H. 重新发送 init 数据，让前端用新状态重建场景
            init_data = generate_send_data(grid_map, agv_manager, ordermanager, data_type="init")
            await websocket.send(json.dumps(init_data))

            NEED_RESET = False
            print("Reset complete.")
            continue

        # -------------------- 主推进循环 --------------------
        while (
            RUNNING
            and not NEED_RESET
            and not ordermanager.is_all_orders_completed()
            and clock.now() < SimConfig.max_steps
        ):
            # pause / resume / step 三种控制都在这里生效
            if not STATE["paused"] or STATE["step_trigger"]:
                simulator.step()
                STATE["step_trigger"] = False

                # 每步向前端推送 update 数据
                step_data = generate_send_data(
                    grid_map,
                    agv_manager,
                    ordermanager,
                    data_type="update"
                )
                await websocket.send(json.dumps(step_data))

            # 处理外部消息（如 damage / repair）
            while not message_queue.empty():
                msg = await message_queue.get()
                fault_manager.handle_message(msg)
                print("Message processed.")

            # 节流，避免事件循环空转
            await asyncio.sleep(0.1)

        # -------------------- 自然结束后的等待态 --------------------
        # 只有在仿真自然结束时才打印最终统计。
        # 如果是用户主动 stop，RUNNING 已经被置为 False，此时不应再输出“等待 reset 或 stop”。
        if RUNNING and not NEED_RESET:
            print("All orders completed or max steps reached; waiting for reset or stop.")
            global_logger.add_runtime_log(global_logger.get_final_metrics(clock.now()))
            print(global_logger.get_final_metrics(clock.now()))

        # 仿真结束后只等待 reset 或 stop
        while RUNNING and not NEED_RESET:
            await asyncio.sleep(0.1)

    print("Simulation loop ended.")


async def ws_handler(websocket):
    """
    WebSocket 消息处理：
    - pause / resume / step / reset / stop 直接修改全局状态
    - 其他命令（如 damage / repair）放入消息队列，交给 simulator_loop 处理
    """
    global RUNNING
    global NEED_RESET

    message_queue = asyncio.Queue()
    sim_task = asyncio.create_task(simulator_loop(websocket, message_queue))

    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
                print("Received message:", msg)
                cmd = msg.get("cmd")

                if cmd == "pause":
                    STATE["paused"] = True

                elif cmd == "resume":
                    STATE["paused"] = False

                elif cmd == "step":
                    STATE["step_trigger"] = True

                elif cmd == "stop":
                    print("Stop command received, exiting...")
                    global_logger.close()
                    RUNNING = False
                    STATE["paused"] = True
                    await websocket.send(json.dumps({"status": "stopping"}))
                    await websocket.close()
                    break

                elif cmd == "reset":
                    print("Reset command received.")
                    NEED_RESET = True

                else:
                    # 其余命令（如 damage / repair）交给仿真循环中的 FaultManager 处理
                    await message_queue.put(msg)

            except Exception as e:
                print("Invalid message:", message, e)

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket closed.")

    finally:
        # 连接关闭时，确保仿真任务被取消，避免孤儿协程
        if not sim_task.done():
            sim_task.cancel()
            try:
                await sim_task
            except asyncio.CancelledError:
                pass
        print("WebSocket handler exited.")


async def main():
    """启动可视化系统：HTTP 服务 + WebSocket 服务 + 自动打开浏览器。"""
    global RUNNING

    http_port = 8000

    # HTTP 静态服务放到后台线程中
    threading.Thread(target=start_http_server, args=(http_port,), daemon=True).start()

    # 自动打开前端页面
    frontend_url = f"http://localhost:{http_port}/frontend/index.html"
    webbrowser.open(frontend_url)
    print(f"Opening browser at {frontend_url}")

    # WebSocket 服务作为前后端实时通信通道
    ws_port = 8765
    async with websockets.serve(ws_handler, "localhost", ws_port):
        print(f"WebSocket server running at ws://localhost:{ws_port}")
        while RUNNING:
            await asyncio.sleep(0.5)

    print("Main loop ended, exiting.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        print("Exit complete.")
