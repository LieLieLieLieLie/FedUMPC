"""Render saved MuJoCo trajectories in isolated OpenGL processes."""

from __future__ import annotations

import argparse

import mujoco

import run_six_algorithms as sim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=sim.METHODS, required=True)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    obstacles = sim.generate_obstacles(args.seed)
    xml = sim.make_scene_xml(obstacles)
    trajectories = sim.load_saved_trajectories(args.method)
    if trajectories is None:
        raise FileNotFoundError(f"No saved trajectory for {args.method}")

    plant = sim.MuJoCoPlant(xml)
    for agent, trajectory in enumerate(trajectories):
        state = trajectory[-1]
        sim.set_playback_state(plant, agent, state, 0.0)
    mujoco.mj_forward(plant.model, plant.data)

    safe = args.method.lower().replace(" ", "_").replace("-", "_")
    destination = sim.FRAMES / f"{safe}.png"
    sim.render_final(plant, trajectories, destination)
    print(f"[MuJoCo] Rendered {args.method}: {destination}")


if __name__ == "__main__":
    main()
