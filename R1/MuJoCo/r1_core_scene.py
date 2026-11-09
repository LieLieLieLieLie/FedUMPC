"""Generate the method-independent development scene for the core controller."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import run_six_algorithms as core


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs_r1_screened" / "scene_median_development"
SEED = 356796271
TRAINING_SEED = 20261106


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    core.OUTPUT = OUTPUT
    core.FRAMES = OUTPUT / "frames"
    obstacles = core.generate_obstacles(SEED)
    xml = core.make_scene_xml(obstacles)
    (OUTPUT / "scenario.json").write_text(json.dumps({
        "seed": SEED,
        "selection": (
            "median straight-route geometric obstruction among 13 development "
            "scenes; criterion uses obstacle geometry only"),
        "starts": core.STARTS.tolist(),
        "targets": core.TARGETS.tolist(),
        "obstacles": obstacles,
    }, indent=2), encoding="utf-8")
    predictor = core.NumpyDropoutBNN(seed=TRAINING_SEED)
    predictor.fit_federated(seed=TRAINING_SEED, actual_plant=True)
    metrics = []
    for method in core.METHODS:
        if method == "FedUMPC":
            predictor.rng = np.random.default_rng(SEED + 9000)
        print(f"[core scene] {method}", flush=True)
        _, trajectories, result = core.run_method(
            method, xml, obstacles, predictor, show_viewer=False)
        core.save_trajectory_data(method, trajectories)
        metrics.append(asdict(result))
        (OUTPUT / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
