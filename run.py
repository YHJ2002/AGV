# run.py
import asyncio
import json
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer
import random
import numpy as np

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

STATE = {
    "paused": True,
    "step_trigger": False,
}
RUNNING = True
NEED_RESET = False
# STATE / RUNNING / NEED_RESET 是运行时全局控制位：
# - paused / step_trigger：控制仿真推进方式（连续运行或单步）
# - RUNNING：进程级退出开关
# - NEED_RESET：异步触发仿真重置

def start_http_server(port=8000):
    """Start local HTTP server (avoids file:// CORS issues)."""
    os.chdir(os.path.abspath("."))
    handler = SimpleHTTPRequestHandler
    with TCPServer(("", port), handler) as httpd:
        print(f"HTTP server running at http://localhost:{port}")
        httpd.serve_forever()


async def simulator_loop(websocket, message_queue):
    global RUNNING
    global NEED_RESET
    print("Simulation begin")

    # -------------------- 1) 核心对象初始化 --------------------
    # 先构建地图/订单/AGV/环境，再构建故障与算法模块。
    # 依赖关系：
    # GridMap -> OrderManager -> AGVManager -> Env -> FaultManager
    grid_map = GridMap()
    ordermanager = OrderManager(grid_map)
    agv_manager = AGVManager(grid_map, ordermanager)
    env = Env(agv_manager, grid_map, ordermanager)
    fault_manager = FaultManager(agv_manager, env, grid_map)

    # 通过工厂按 SimConfig 选择调度器与路径规划器
    scheduler = build_scheduler(env, agv_manager, ordermanager, grid_map, fault_manager)
    planner = build_planner(env, agv_manager, ordermanager, grid_map, fault_manager)

    simulator = Simulator(grid_map, agv_manager, ordermanager, env, scheduler, planner)

    # 首帧发送 init 数据，前端据此创建地图、AGV、货架等对象
    init_data = generate_send_data(grid_map, agv_manager, ordermanager, data_type="init")
    await websocket.send(json.dumps(init_data))

    # -------------------- 2) 仿真生命周期循环 --------------------
    while RUNNING:
        # reset 是“热重置”：不重启进程，重置时钟、环境与算法状态
        if NEED_RESET:
            print("Resetting simulation...")
            clock.reset()
            env.reset()
            scheduler.reset()
            global_logger.reset()
            NEED_RESET = False
            print("Reset complete.")
            continue
        # 主推进循环：当未结束、未重置且未超最大步时执行
        while (RUNNING
               and not NEED_RESET
               and not ordermanager.is_all_orders_completed()
               and clock.now() < SimConfig.max_steps):

            # pause/resume/step 三种控制都在这里生效
            if not STATE["paused"] or STATE["step_trigger"]:
                simulator.step()
                STATE["step_trigger"] = False
                # 每步都推送 update，让前端刷新 AGV/箱体/指标/订单面板
                step_data = generate_send_data(grid_map, agv_manager, ordermanager, data_type="update")
                await websocket.send(json.dumps(step_data))

            # 处理运行中外部消息（如故障注入/修复）
            while not message_queue.empty():
                msg = await message_queue.get()
                fault_manager.handle_message(msg)
                print("Message processed.")

            # 节流，避免事件循环空转
            await asyncio.sleep(0.1)

        if not NEED_RESET:
            print("All orders completed or max steps reached; waiting for reset or stop.")
            global_logger.add_runtime_log(global_logger.get_final_metrics(clock.now()))
            print(global_logger.get_final_metrics(clock.now()))

        # 仿真完成后进入等待态，只响应 reset 或 stop
        while RUNNING and not NEED_RESET:
            await asyncio.sleep(0.1)

    print("Simulation loop ended.")


async def ws_handler(websocket):
    global RUNNING
    # message_queue：将“非核心控制命令”转交给 simulator_loop 处理
    message_queue = asyncio.Queue()
    sim_task = asyncio.create_task(simulator_loop(websocket, message_queue))

    try:
        async for message in websocket:
            try:
                msg = json.loads(message)
                print("Received message:", msg)
                cmd = msg.get("cmd")
                # 控制命令直接作用于全局状态
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
                    global NEED_RESET
                    # reset 延迟到 simulator_loop 主循环统一执行
                    NEED_RESET = True
                    # await websocket.send(json.dumps({"status": "resetting"}))
                    # continue
                else:
                    # 其余命令（如 damage/repair）入队交给 FaultManager
                    await message_queue.put(msg)

            except Exception as e:
                print("Invalid message:", message, e)

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket closed.")

    finally:
        # 连接断开时，确保仿真任务被取消，避免孤儿协程
        if not sim_task.done():
            sim_task.cancel()
            try:
                await sim_task
            except asyncio.CancelledError:
                pass
        print("WebSocket handler exited.")


async def main():
    """Start visualization: HTTP server + WebSocket + open browser."""
    global RUNNING
    http_port = 8000
    # HTTP 静态服务放在线程中，主协程专注 WebSocket 与生命周期控制
    threading.Thread(target=start_http_server, args=(http_port,), daemon=True).start()


    # 自动打开前端页面
    frontend_url = f"http://localhost:{http_port}/frontend/index.html"
    webbrowser.open(frontend_url)
    print(f"Opening browser at {frontend_url}")

    ws_port = 8765
    # WebSocket 服务作为前后端唯一实时控制/数据通道
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
