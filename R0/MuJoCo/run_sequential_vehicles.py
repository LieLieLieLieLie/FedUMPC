"""Sequential four-vehicle MuJoCo demonstration.

Vehicle A completes its route before B starts, followed by C and D.  The
controller, vehicle plant, obstacle map, and target definitions are reused from
the original MuJoCo experiment, while the launch schedule is intentionally
changed for a collision-free visual demonstration.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
# The retained MuJoCo package is self-contained: the common controller and
# plant implementation lives beside this sequential demonstration.
sys.path.insert(0, str(ROOT))

import run_six_algorithms as core


OUTPUT = ROOT / "outputs"
HANDOFF_HOLD_STEPS = 10  # 1.0 s between two vehicles in the demonstration
VEHICLE_NAMES = ("A", "B", "C", "D")


def safe_method_name(method: str) -> str:
    return method.lower().replace(" ", "_").replace("-", "_")


def timeline_path(method: str) -> Path:
    return OUTPUT / f"{safe_method_name(method)}_sequential_timeline.json"


def metrics_path(method: str) -> Path:
    return OUTPUT / f"{safe_method_name(method)}_sequential_metrics.json"


def freeze_vehicle(plant: core.MuJoCoPlant, agent: int):
    plant.data.qvel[plant.qvel_addresses[agent]] = 0.0
    plant.data.ctrl[plant.ctrl_ids[agent]] = 0.0


def activate_vehicle(plant: core.MuJoCoPlant, agent: int):
    yaw = float(plant.state(agent)[3])
    speed = float(core.STARTS[agent, 2])
    addresses = plant.qvel_addresses[agent]
    plant.data.qvel[addresses[0]] = speed * math.cos(yaw)
    plant.data.qvel[addresses[1]] = speed * math.sin(yaw)
    plant.data.qvel[addresses[2]] = 0.0
    mujoco.mj_forward(plant.model, plant.data)


def snapshot(plant: core.MuJoCoPlant) -> np.ndarray:
    return np.stack([plant.state(agent) for agent in range(4)], axis=0)


def run_sequential_simulation(
    method: str,
    seed: int,
    skip_bnn_training: bool,
) -> tuple[str, np.ndarray, list[dict[str, object]]]:
    obstacles = core.generate_obstacles(seed)
    xml = core.make_scene_xml(obstacles)
    (ROOT / f"scene_seed{seed}.xml").write_text(xml, encoding="utf-8")

    bnn = core.NumpyDropoutBNN(seed=seed)
    if not skip_bnn_training:
        print(
            "[MuJoCo] Training the lightweight proximal federated "
            "MC-dropout dynamics model ...",
            flush=True,
        )
        bnn.fit_federated(seed=seed)

    plant = core.MuJoCoPlant(xml)
    for agent in range(4):
        freeze_vehicle(plant, agent)
    mujoco.mj_forward(plant.model, plant.data)

    controllers = [core.SamplingMPC(method, obstacles, bnn) for _ in range(4)]
    controls = [np.zeros(2, dtype=float) for _ in range(4)]
    timeline = [snapshot(plant)]
    results: list[dict[str, object]] = []

    for agent in range(4):
        label = VEHICLE_NAMES[agent]
        print(
            f"[MuJoCo] Vehicle {label} starts; vehicles after {label} remain "
            "stationary.",
            flush=True,
        )
        activate_vehicle(plant, agent)
        success = False
        collision = False
        clearances: list[float] = []
        steps = 0

        for _ in range(core.MAX_STEPS):
            previous_state = plant.state(agent)
            controls[agent] = controllers[agent].action(
                previous_state,
                core.TARGETS[agent],
            )
            active = [index == agent for index in range(4)]
            plant.step(controls, active)

            # Waiting and completed vehicles are fixed exactly in place.
            for other in range(4):
                if other != agent:
                    freeze_vehicle(plant, other)
            mujoco.mj_forward(plant.model, plant.data)

            current_state = plant.state(agent)
            controllers[agent].update_model(
                previous_state,
                controls[agent],
                current_state,
            )
            clearances.append(
                float(core.min_signed_distance(current_state[:2], obstacles)[0])
            )
            steps += 1
            timeline.append(snapshot(plant))

            if core.is_collision(plant, agent):
                collision = True
                break
            if core.reached_goal(current_state, agent):
                success = True
                break

        freeze_vehicle(plant, agent)
        mujoco.mj_forward(plant.model, plant.data)
        for _ in range(HANDOFF_HOLD_STEPS):
            timeline.append(snapshot(plant))

        result = {
            "vehicle": label,
            "agent_index": agent,
            "success": success,
            "static_obstacle_collision": collision,
            "steps": steps,
            "min_clearance": float(min(clearances)) if clearances else 0.0,
            "final_state": plant.state(agent).tolist(),
        }
        results.append(result)
        outcome = "reached the goal" if success else (
            "hit a static obstacle" if collision else "timed out"
        )
        print(
            f"[MuJoCo] Vehicle {label} {outcome} after {steps} control steps. "
            f"Next vehicle may start.",
            flush=True,
        )

    return xml, np.asarray(timeline, dtype=float), results


def save_run(
    method: str,
    seed: int,
    timeline: np.ndarray,
    results: list[dict[str, object]],
):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": method,
        "seed": seed,
        "control_step": core.DT_CONTROL,
        "physics_step": core.DT_PHYSICS,
        "schedule": "A -> B -> C -> D",
        "timeline": timeline.tolist(),
    }
    timeline_path(method).write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    metrics_path(method).write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )


def load_cached_run(method: str, seed: int) -> np.ndarray | None:
    path = timeline_path(method)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("method") != method or int(payload.get("seed", -1)) != seed:
        return None
    timeline = np.asarray(payload.get("timeline", []), dtype=float)
    if timeline.ndim != 3 or timeline.shape[1:] != (4, 4) or len(timeline) < 2:
        return None
    return timeline


def set_timeline_frame(
    plant: core.MuJoCoPlant,
    state_frame: np.ndarray,
    yaw_rates: np.ndarray,
):
    for agent in range(4):
        core.set_playback_state(
            plant,
            agent,
            state_frame[agent],
            float(yaw_rates[agent]),
        )
    mujoco.mj_forward(plant.model, plant.data)


def play_timeline(
    method: str,
    xml: str,
    timeline: np.ndarray,
    initial_speed: float = 1.0,
):
    import mujoco.viewer

    plant = core.MuJoCoPlant(xml)
    playback = {
        "restart": False,
        "paused": False,
        "speed": float(np.clip(initial_speed, 0.25, 8.0)),
    }

    def key_callback(keycode: int):
        if keycode in (ord("R"), ord("r")):
            playback["restart"] = True
            playback["paused"] = False
            print("[MuJoCo] Restart requested (R).", flush=True)
        elif keycode == 32:
            playback["paused"] = not playback["paused"]
            status = "paused" if playback["paused"] else "resumed"
            print(f"[MuJoCo] Playback {status} (Space).", flush=True)
        elif keycode in (ord("="), 334):
            playback["speed"] = min(8.0, playback["speed"] * 2.0)
            print(
                f"[MuJoCo] Playback speed: {playback['speed']:g}x (+).",
                flush=True,
            )
        elif keycode in (ord("-"), 333):
            playback["speed"] = max(0.25, playback["speed"] / 2.0)
            print(
                f"[MuJoCo] Playback speed: {playback['speed']:g}x (-).",
                flush=True,
            )

    with mujoco.viewer.launch_passive(
        plant.model,
        plant.data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
        viewer.cam.distance = 58.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -72.0
        print(
            f"[MuJoCo] Sequential {method} viewer opened. "
            f"Order: A -> B -> C -> D. Initial speed: {playback['speed']:g}x. "
            "Press +/- to change speed, R to replay and Space to pause.",
            flush=True,
        )

        while viewer.is_running():
            playback["restart"] = False
            playback["paused"] = False
            with viewer.lock():
                set_timeline_frame(plant, timeline[0], np.zeros(4))
            viewer.sync()

            for frame_index in range(len(timeline) - 1):
                if playback["restart"]:
                    break
                state0 = timeline[frame_index]
                state1 = timeline[frame_index + 1]
                yaw_delta = np.asarray(
                    core.wrap_angle(state1[:, 3] - state0[:, 3]),
                    dtype=float,
                )
                for substep in range(core.SUBSTEPS):
                    if not viewer.is_running():
                        return
                    while (
                        playback["paused"]
                        and not playback["restart"]
                        and viewer.is_running()
                    ):
                        viewer.sync()
                        time.sleep(0.05)
                    if playback["restart"]:
                        break

                    frame_start = time.perf_counter()
                    alpha = (substep + 1) / core.SUBSTEPS
                    state = (1.0 - alpha) * state0 + alpha * state1
                    state[:, 3] = state0[:, 3] + alpha * yaw_delta
                    with viewer.lock():
                        set_timeline_frame(
                            plant,
                            state,
                            yaw_delta / core.DT_CONTROL,
                        )
                    viewer.sync()
                    frame_period = core.DT_PHYSICS / playback["speed"]
                    remaining = frame_period - (
                        time.perf_counter() - frame_start
                    )
                    if remaining > 0.0:
                        time.sleep(remaining)

            if playback["restart"]:
                print("[MuJoCo] Replaying from vehicle A ...", flush=True)
                continue

            print(
                "[MuJoCo] Sequential playback finished. Press R to replay "
                "or close the window to exit.",
                flush=True,
            )
            while viewer.is_running() and not playback["restart"]:
                viewer.sync()
                time.sleep(0.05)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=core.METHODS,
        default="FedUMPC",
        help="Controller used by all four sequential vehicles",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Re-run the sequential MuJoCo physics instead of using the cache",
    )
    parser.add_argument(
        "--skip-bnn-training",
        action="store_true",
        help="Diagnostic only; do not use for formal results",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Compute/save the timeline without opening the viewer",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Viewer playback speed multiplier (0.25 to 8, default: 1)",
    )
    args = parser.parse_args()
    if not 0.25 <= args.speed <= 8.0:
        parser.error("--speed must be between 0.25 and 8")
    return args


def main():
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    obstacles = core.generate_obstacles(args.seed)
    xml = core.make_scene_xml(obstacles)
    (ROOT / f"scene_seed{args.seed}.xml").write_text(xml, encoding="utf-8")

    timeline = None if args.recompute else load_cached_run(args.method, args.seed)
    if timeline is None:
        _, timeline, results = run_sequential_simulation(
            args.method,
            args.seed,
            args.skip_bnn_training,
        )
        save_run(args.method, args.seed, timeline, results)
        print(
            f"[MuJoCo] Saved {len(timeline)} sequential control frames to "
            f"{timeline_path(args.method)}",
            flush=True,
        )
    else:
        print(
            f"[MuJoCo] Loaded cached sequential timeline: "
            f"{timeline_path(args.method)}",
            flush=True,
        )

    if not args.no_viewer:
        play_timeline(
            args.method,
            xml,
            timeline,
            initial_speed=args.speed,
        )


if __name__ == "__main__":
    main()
