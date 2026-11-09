"""Export actual MuJoCo body-state trajectories for the predeclared first scene."""

from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

import r1_mujoco_final as frozen
import run_six_algorithms as core


def main():
    frozen.freeze_or_verify_protocol()
    seed = frozen.SEEDS[0]
    output = frozen.OUTPUT / "scene_first_heldout"
    output.mkdir(parents=True, exist_ok=True)
    obstacles = core.generate_obstacles(seed)
    xml = core.make_scene_xml(obstacles)
    model = core.NumpyDropoutBNN(seed=frozen.TRAINING_SEED)
    model.fit_federated(seed=frozen.TRAINING_SEED, actual_plant=True)
    metrics = []
    for method in core.METHODS:
        if method == "FedUMPC":
            model.rng = np.random.default_rng(seed + 9000)
        _, trajectories, result = core.run_method(
            method, xml, obstacles, model, show_viewer=False)
        safe = method.lower().replace(" ", "_").replace("-", "_")
        payload = {f"agent_{i + 1}": trajectory.tolist()
                   for i, trajectory in enumerate(trajectories)}
        (output / f"{safe}_trajectories.json").write_text(
            json.dumps(payload), encoding="utf-8")
        metrics.append(asdict(result))
    scenario = {
        "seed": seed,
        "starts": [list(value) for value in core.STARTS],
        "targets": [list(value) for value in core.TARGETS],
        "obstacles": obstacles,
    }
    (output / "scenario.json").write_text(
        json.dumps(scenario, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote actual held-out trajectories to {output}")


if __name__ == "__main__":
    main()
