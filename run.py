# run.py
import asyncio
import json
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer

import websockets

from config.settings import OrderMode, PlannerType, SchedulerType, SimConfig
from core.agvmanager import AGVManager
from core.data_generator import generate_send_data
from core.env import Env
from core.fault_manager import FaultManager
from core.gridmap import GridMap
from core.ordermanager import OrderManager
from core.simulator import Simulator
from utils.algorithm_factory import build_planner, build_scheduler
from utils.logger import global_logger
from utils.simulation_clock import clock

STATE = {
    "paused": False,
    "step_trigger": False,
}
RUNNING = True
NEED_RESET = False


def serialize_algorithm_config():
    return {
        "scheduler": SimConfig.scheduler_type.value,
        "planner": SimConfig.planner_type.value,
        "order_mode": SimConfig.order_mode.value,
    }


def build_algorithm_config_message(status, message, applies_on_reset=True):
    return {
        "type": "algorithm_config",
        "status": status,
        "message": message,
        "applies_on_reset": applies_on_reset,
        **serialize_algorithm_config(),
    }


def apply_algorithm_config(msg):
    scheduler_value = msg.get("scheduler")
    planner_value = msg.get("planner")
    order_mode_value = msg.get("order_mode")

    missing_fields = [
        key for key, value in (
            ("scheduler", scheduler_value),
            ("planner", planner_value),
            ("order_mode", order_mode_value),
        ) if value is None
    ]
    if missing_fields:
        return False, build_algorithm_config_message(
            "error",
            f"Missing fields: {', '.join(missing_fields)}",
        )

    try:
        SimConfig.scheduler_type = SchedulerType(scheduler_value)
        SimConfig.planner_type = PlannerType(planner_value)
        SimConfig.order_mode = OrderMode(order_mode_value)
    except ValueError as exc:
        return False, build_algorithm_config_message("error", str(exc))

    global_logger.add_runtime_log(
        "[Config] Updated algorithms: "
        f"scheduler={SimConfig.scheduler_type.value}, "
        f"planner={SimConfig.planner_type.value}, "
        f"order_mode={SimConfig.order_mode.value}. "
        "Reset to apply."
    )

    return True, build_algorithm_config_message(
        "ok",
        "Algorithm configuration saved. Reset to apply.",
    )


def start_http_server(port=8000):
    """Start a local static HTTP server for the frontend."""
    os.chdir(os.path.abspath("."))
    handler = SimpleHTTPRequestHandler
    with TCPServer(("", port), handler) as httpd:
        print(f"HTTP server running at http://localhost:{port}")
        httpd.serve_forever()


async def send_init_payload(websocket, grid_map, agv_manager, ordermanager, message):
    init_data = generate_send_data(grid_map, agv_manager, ordermanager, data_type="init")
    init_data["algorithm_config"] = build_algorithm_config_message(
        "active",
        message,
    )
    await websocket.send(json.dumps(init_data))


async def simulator_loop(websocket, message_queue):
    """Run the simulation lifecycle, including resets and step updates."""
    global RUNNING
    global NEED_RESET

    print("Simulation begin")

    grid_map = GridMap()
    ordermanager = OrderManager(grid_map)
    agv_manager = AGVManager(grid_map, ordermanager)
    env = Env(agv_manager, grid_map, ordermanager)
    fault_manager = FaultManager(agv_manager, env, grid_map)

    scheduler = build_scheduler(env, agv_manager, ordermanager, grid_map, fault_manager)
    planner = build_planner(env, agv_manager, ordermanager, grid_map, fault_manager)
    simulator = Simulator(grid_map, agv_manager, ordermanager, env, scheduler, planner)
    global_logger.record_agv_positions(clock.now(), agv_manager)
    paths_exported_for_cycle = False

    await send_init_payload(
        websocket,
        grid_map,
        agv_manager,
        ordermanager,
        "Current active algorithms.",
    )

    while RUNNING:
        if NEED_RESET:
            print("Resetting simulation...")

            STATE["paused"] = False
            STATE["step_trigger"] = False

            await websocket.send(json.dumps({"type": "reset"}))
            if not paths_exported_for_cycle:
                global_logger.export_agv_paths(grid_map.width, grid_map.height)
                paths_exported_for_cycle = True

            clock.reset()
            global_logger.reset()

            grid_map.reset_map()
            ordermanager.reset_order()
            agv_manager.reset_agvs()
            fault_manager.reset()
            env.reset()

            scheduler = build_scheduler(env, agv_manager, ordermanager, grid_map, fault_manager)
            planner = build_planner(env, agv_manager, ordermanager, grid_map, fault_manager)
            simulator = Simulator(grid_map, agv_manager, ordermanager, env, scheduler, planner)
            global_logger.record_agv_positions(clock.now(), agv_manager)

            while not message_queue.empty():
                _ = await message_queue.get()

            await send_init_payload(
                websocket,
                grid_map,
                agv_manager,
                ordermanager,
                "Algorithms applied after reset.",
            )

            NEED_RESET = False
            paths_exported_for_cycle = False
            print("Reset complete.")
            continue

        while (
            RUNNING
            and not NEED_RESET
            and not ordermanager.is_all_orders_completed()
            and clock.now() < SimConfig.max_steps
        ):
            if not STATE["paused"] or STATE["step_trigger"]:
                simulator.step()
                STATE["step_trigger"] = False

                step_data = generate_send_data(
                    grid_map,
                    agv_manager,
                    ordermanager,
                    data_type="update",
                )
                await websocket.send(json.dumps(step_data))

            while not message_queue.empty():
                msg = await message_queue.get()
                fault_manager.handle_message(msg)
                print("Message processed.")

            await asyncio.sleep(0.1)

        if not NEED_RESET:
            if not paths_exported_for_cycle:
                global_logger.export_agv_paths(grid_map.width, grid_map.height)
                paths_exported_for_cycle = True
            print("All orders completed or max steps reached; stopping automatically.")
            global_logger.add_runtime_log(global_logger.get_final_metrics(clock.now()))
            print(global_logger.get_final_metrics(clock.now()))
            STATE["paused"] = True
            RUNNING = False
            await websocket.send(json.dumps({
                "type": "simulation_complete",
                "message": "Simulation complete. AGV path files exported to logs/."
            }))
            await websocket.close()
            break

    print("Simulation loop ended.")
    if not paths_exported_for_cycle:
        global_logger.export_agv_paths(grid_map.width, grid_map.height)


async def ws_handler(websocket):
    """
    Handle frontend WebSocket commands.
    Control commands act immediately; other messages are passed to the sim loop.
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

                elif cmd == "set_algorithms":
                    _, payload = apply_algorithm_config(msg)
                    await websocket.send(json.dumps(payload))

                else:
                    await message_queue.put(msg)

            except Exception as exc:
                print("Invalid message:", message, exc)

    except websockets.exceptions.ConnectionClosed:
        print("WebSocket closed.")

    finally:
        if not sim_task.done():
            sim_task.cancel()
            try:
                await sim_task
            except asyncio.CancelledError:
                pass
        print("WebSocket handler exited.")


async def main():
    """Start the HTTP server, WebSocket server, and open the frontend."""
    global RUNNING

    http_port = 8000
    threading.Thread(target=start_http_server, args=(http_port,), daemon=True).start()

    frontend_url = f"http://localhost:{http_port}/frontend/index.html"
    webbrowser.open(frontend_url)
    print(f"Opening browser at {frontend_url}")

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
