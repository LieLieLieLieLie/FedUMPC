"""Auditable MuJoCo tuning sweep for FedUMPC deployment parameters.

The sweep keeps the scene, BNN, candidate set, horizon, and physical plant
fixed.  It changes only the documented FedUMPC cost/margin parameters and
ranks configurations lexicographically by success, collision avoidance,
minimum clearance, mean completion steps, and mean path length.
"""

from __future__ import annotations

import copy
import inspect
import json
import os
import random as py_random
import sys
import types
from pathlib import Path

# Install deterministic fallbacks before NumPy/MuJoCo are imported on the
# experiment workstation, whose Windows random/overlapped providers can fail.
try:
    os.urandom(1)
except OSError:
    _fallback_rng = py_random.Random(20261103)

    def _deterministic_urandom(n):
        return bytes(_fallback_rng.getrandbits(8) for _ in range(n))

    os.urandom = _deterministic_urandom
    py_random._urandom = _deterministic_urandom

try:
    import _overlapped  # noqa: F401
except OSError:
    _asyncio_stub = types.ModuleType("asyncio")
    _asyncio_stub.iscoroutinefunction = inspect.iscoroutinefunction
    _asyncio_stub.coroutines = types.SimpleNamespace(_is_coroutine=object())
    sys.modules["asyncio"] = _asyncio_stub

import numpy as np

import run_six_algorithms as sim


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
DEVELOPMENT_SEEDS = (11, 23)


CONFIGS = [
    # A finite, preregistered-style grid evaluated only on development scenes.
    # Test seeds in r1_multiseed.py are never consulted by this selection.
    dict(name="c01", path_weight=0, northbound_path_weight=0,
         margin_offset=0.00, margin_scale=0.30, margin_cap=1.20,
         soft_weight=250, mismatch_weight=3, uncertainty_weight=8),
    dict(name="c02", path_weight=40, northbound_path_weight=40,
         margin_offset=0.00, margin_scale=0.30, margin_cap=1.20,
         soft_weight=250, mismatch_weight=3, uncertainty_weight=12),
    dict(name="c03", path_weight=40, northbound_path_weight=40,
         margin_offset=0.00, margin_scale=0.50, margin_cap=1.40,
         soft_weight=350, mismatch_weight=3, uncertainty_weight=12),
    dict(name="c04", path_weight=40, northbound_path_weight=40,
         margin_offset=0.15, margin_scale=0.45, margin_cap=1.40,
         soft_weight=350, mismatch_weight=3, uncertainty_weight=12),
    dict(name="c05", path_weight=120, northbound_path_weight=120,
         margin_offset=0.00, margin_scale=0.45, margin_cap=1.40,
         soft_weight=350, mismatch_weight=3, uncertainty_weight=12),
    dict(name="c06", path_weight=120, northbound_path_weight=300,
         margin_offset=0.10, margin_scale=0.60, margin_cap=1.50,
         soft_weight=400, mismatch_weight=5, uncertainty_weight=12),
    dict(name="c07", path_weight=40, northbound_path_weight=40,
         margin_offset=0.00, margin_scale=0.70, margin_cap=1.50,
         soft_weight=400, mismatch_weight=5, uncertainty_weight=12),
    dict(name="c08", path_weight=40, northbound_path_weight=40,
         margin_offset=0.20, margin_scale=0.30, margin_cap=1.20,
         soft_weight=350, mismatch_weight=3, uncertainty_weight=8),
    dict(name="c09", path_weight=0, northbound_path_weight=0,
         margin_offset=0.00, margin_scale=0.50, margin_cap=1.40,
         soft_weight=250, mismatch_weight=5, uncertainty_weight=20),
    dict(name="c10", path_weight=120, northbound_path_weight=300,
         margin_offset=0.00, margin_scale=0.45, margin_cap=1.40,
         soft_weight=350, mismatch_weight=5, uncertainty_weight=8),
]


def apply_config(config: dict[str, float | str]) -> None:
    sim.FED_PATH_WEIGHT = float(config["path_weight"])
    sim.FED_PATH_WEIGHT_NORTHBOUND = float(config["northbound_path_weight"])
    sim.FED_MARGIN_OFFSET = float(config["margin_offset"])
    sim.FED_MARGIN_SCALE = float(config["margin_scale"])
    sim.FED_MARGIN_CAP = float(config["margin_cap"])
    sim.FED_SOFT_WEIGHT = float(config["soft_weight"])
    sim.FED_MISMATCH_WEIGHT = float(config["mismatch_weight"])
    sim.FED_UNCERTAINTY_WEIGHT = float(config["uncertainty_weight"])


def path_statistics(trajectories: list[np.ndarray]) -> tuple[float, float]:
    lengths = [
        float(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1).sum())
        for trajectory in trajectories
    ]
    return float(np.mean(lengths)), float(np.max(lengths))


def main() -> None:
    checkpoint = OUTPUT / "fedumpc_tuning_checkpoint.json"
    raw_records = (json.loads(checkpoint.read_text(encoding="utf-8"))
                   if checkpoint.exists() else [])
    scenes = {}
    for seed in DEVELOPMENT_SEEDS:
        obstacles = sim.generate_obstacles(seed)
        xml = sim.make_scene_xml(obstacles)
        model = sim.NumpyDropoutBNN(seed=seed)
        print(f"[tuning] Training development BNN | seed={seed}", flush=True)
        model.fit_federated(seed=seed)
        scenes[seed] = (obstacles, xml, model)

    aggregate_records = []
    for index, config in enumerate(CONFIGS, start=1):
        apply_config(config)
        print(f"[tuning] {index:02d}/{len(CONFIGS):02d} {config['name']}", flush=True)
        config_records = []
        for seed, (obstacles, xml, base_bnn) in scenes.items():
            previous = next((row for row in raw_records
                             if row["name"] == config["name"]
                             and row["seed"] == seed), None)
            if previous is not None:
                config_records.append(previous)
                print(f"  seed={seed} restored checkpoint", flush=True)
                continue
            bnn = copy.deepcopy(base_bnn)
            bnn.rng = np.random.default_rng(seed + 9000)
            _, trajectories, metrics = sim.run_method(
                "FedUMPC", xml, obstacles, bnn, show_viewer=False)
            mean_path, max_path = path_statistics(trajectories)
            record = {"seed": seed, **config, **metrics.__dict__,
                      "mean_path_m": mean_path, "max_path_m": max_path}
            raw_records.append(record)
            checkpoint.write_text(json.dumps(raw_records, indent=2),
                                  encoding="utf-8")
            config_records.append(record)
            print(
                f"  seed={seed} success={metrics.success}/4 "
                f"collision={metrics.collisions}/4 "
                f"viol={metrics.violation_frequency_pct:.2f}% "
                f"min_clearance={metrics.min_clearance:.3f} m",
                flush=True,
            )
        aggregate_records.append({
            **config,
            "success": sum(row["success"] for row in config_records),
            "collisions": sum(row["collisions"] for row in config_records),
            "violation_frequency_pct": float(np.mean(
                [row["violation_frequency_pct"] for row in config_records])),
            "min_clearance": min(row["min_clearance"] for row in config_records),
            "mean_steps": float(np.mean([row["mean_steps"] for row in config_records])),
            "mean_path_m": float(np.mean([row["mean_path_m"] for row in config_records])),
        })

    aggregate_records.sort(
        key=lambda row: (
            -row["success"],
            row["collisions"],
            row["violation_frequency_pct"],
            -row["min_clearance"],
            row["mean_steps"],
            row["mean_path_m"],
        )
    )
    destination = OUTPUT / "fedumpc_tuning_development.json"
    destination.write_text(json.dumps({
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "ranking": aggregate_records,
        "raw": raw_records,
    }, indent=2), encoding="utf-8")
    print(f"[tuning] Best configuration: {aggregate_records[0]['name']}")
    print(f"[tuning] Audit saved to {destination}")


if __name__ == "__main__":
    main()
