"""Auditable MuJoCo tuning sweep for FedUMPC deployment parameters.

The sweep keeps the scene, BNN, candidate set, horizon, and physical plant
fixed.  It changes only the documented FedUMPC cost/margin parameters and
ranks configurations lexicographically by success, collision avoidance,
minimum clearance, mean completion steps, and mean path length.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

import run_six_algorithms as sim


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
SEED = 123


CONFIGS = [
    {
        "name": "legacy_special_case",
        "path_weight": 120.0,
        "northbound_path_weight": 6000.0,
        "margin_offset": 0.56,
        "margin_scale": 1.15,
        "margin_cap": 1.60,
        "soft_weight": 520.0,
    },
]

for path_weight, northbound_path_weight in [
    (0.0, 0.0),
    (40.0, 40.0),
    (120.0, 120.0),
    (120.0, 300.0),
]:
    for margin_offset, margin_scale, margin_cap, soft_weight in [
        (0.32, 0.80, 1.20, 420.0),
        (0.42, 0.90, 1.40, 470.0),
        (0.50, 1.00, 1.50, 520.0),
    ]:
        CONFIGS.append(
            {
                "name": (
                    f"pw{path_weight:g}_npw{northbound_path_weight:g}_"
                    f"mo{margin_offset:.2f}_ms{margin_scale:.2f}"
                ),
                "path_weight": path_weight,
                "northbound_path_weight": northbound_path_weight,
                "margin_offset": margin_offset,
                "margin_scale": margin_scale,
                "margin_cap": margin_cap,
                "soft_weight": soft_weight,
            }
        )


def apply_config(config: dict[str, float | str]) -> None:
    sim.FED_PATH_WEIGHT = float(config["path_weight"])
    sim.FED_PATH_WEIGHT_NORTHBOUND = float(config["northbound_path_weight"])
    sim.FED_MARGIN_OFFSET = float(config["margin_offset"])
    sim.FED_MARGIN_SCALE = float(config["margin_scale"])
    sim.FED_MARGIN_CAP = float(config["margin_cap"])
    sim.FED_SOFT_WEIGHT = float(config["soft_weight"])


def path_statistics(trajectories: list[np.ndarray]) -> tuple[float, float]:
    lengths = [
        float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())
        for trajectory in trajectories
    ]
    return float(np.mean(lengths)), float(np.max(lengths))


def main() -> None:
    obstacles = sim.generate_obstacles(SEED)
    xml = sim.make_scene_xml(obstacles)
    base_bnn = sim.NumpyDropoutBNN(seed=SEED)
    print("[tuning] Training the shared BNN once for all deterministic configurations ...", flush=True)
    base_bnn.fit_federated(seed=SEED)

    records = []
    for index, config in enumerate(CONFIGS, start=1):
        apply_config(config)
        bnn = copy.deepcopy(base_bnn)
        bnn.rng = np.random.default_rng(SEED + 9000)
        print(f"[tuning] {index:02d}/{len(CONFIGS):02d} {config['name']}", flush=True)
        _, trajectories, metrics = sim.run_method(
            "FedUMPC",
            xml,
            obstacles,
            bnn,
            show_viewer=False,
        )
        mean_path, max_path = path_statistics(trajectories)
        record = {
            **config,
            **metrics.__dict__,
            "mean_path_m": mean_path,
            "max_path_m": max_path,
        }
        records.append(record)
        print(
            "  "
            f"success={metrics.success}/4 collision={metrics.collisions}/4 "
            f"min_clearance={metrics.min_clearance:.3f} m "
            f"mean_steps={metrics.mean_steps:.1f} mean_path={mean_path:.2f} m",
            flush=True,
        )

    records.sort(
        key=lambda row: (
            -row["success"],
            row["collisions"],
            -row["min_clearance"],
            row["mean_steps"],
            row["mean_path_m"],
        )
    )
    destination = OUTPUT / "fedumpc_tuning_seed123.json"
    destination.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[tuning] Best configuration: {records[0]['name']}")
    print(f"[tuning] Audit saved to {destination}")


if __name__ == "__main__":
    main()
