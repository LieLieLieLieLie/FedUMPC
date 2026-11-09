"""Run the revised MuJoCo benchmark over independent scene seeds.

The script is checkpointed at the (seed, method) level and writes both raw
results and seed-level descriptive summaries.  It deliberately disables
rendering because the paper figure continues to use the audited seed-123 scene;
the additional seeds quantify generalizability rather than select a new view.
"""

from __future__ import annotations

import csv
import inspect
import json
import os
import random as py_random
import sys
import types
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

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

import run_six_algorithms as core


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
RAW_PATH = OUTPUT / "r1_mujoco_multiseed.csv"
SUMMARY_PATH = OUTPUT / "r1_mujoco_multiseed_summary.csv"
SEEDS = (42, 123, 456, 789, 1024)


def existing_keys() -> set[tuple[int, str]]:
    if not RAW_PATH.exists():
        return set()
    with RAW_PATH.open(newline="", encoding="utf-8-sig") as stream:
        return {(int(row["seed"]), row["method"])
                for row in csv.DictReader(stream)}


def append_row(row: dict) -> None:
    write_header = not RAW_PATH.exists()
    with RAW_PATH.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary() -> None:
    with RAW_PATH.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    numeric = [
        "success", "collisions", "mean_clearance", "min_clearance",
        "mean_steps", "violation_frequency_pct", "violation_episodes",
        "mean_violation_depth", "max_violation_depth",
        "max_violation_duration_s",
    ]
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summary_rows = []
    for method in core.METHODS:
        values = grouped.get(method, [])
        if not values:
            continue
        result = {"method": method, "seeds": len(values)}
        for name in numeric:
            array = np.asarray([float(row[name]) for row in values])
            result[f"{name}_mean"] = float(array.mean())
            result[f"{name}_sample_sd"] = (
                float(array.std(ddof=1)) if len(array) > 1 else 0.0)
        summary_rows.append(result)
    if not summary_rows:
        return
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    done = existing_keys()
    for seed in SEEDS:
        obstacles = core.generate_obstacles(seed)
        xml = core.make_scene_xml(obstacles)
        model = core.NumpyDropoutBNN(seed=seed)
        print(f"[MuJoCo R1] training BNN | seed={seed}", flush=True)
        model.fit_federated(seed=seed)
        model_dir = OUTPUT / 'models'
        model_dir.mkdir(exist_ok=True)
        np.savez(model_dir / f'numpy_predictor_seed{seed}.npz',
                 **model.params, input_mean=model.input_mean,
                 input_std=model.input_std, output_mean=model.output_mean,
                 output_std=model.output_std)
        for method in core.METHODS:
            if (seed, method) in done:
                continue
            if method == "FedUMPC":
                model.rng = np.random.default_rng(seed + 9000)
            print(f"[MuJoCo R1] {method} | seed={seed}", flush=True)
            _, _, metrics = core.run_method(
                method, xml, obstacles, model, show_viewer=False)
            append_row({"seed": seed, **asdict(metrics)})
            write_summary()
    metadata = {
        "seeds": list(SEEDS),
        "safety_margin_m": core.SAFETY_MARGIN,
        "clearance_geometry": (
            "vehicle reference-point center to circular-obstacle boundary"
        ),
        "contact_geometry": "MuJoCo vehicle-body to obstacle-geom contact",
        "control_period_s": core.DT_CONTROL,
    }
    (OUTPUT / "r1_mujoco_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {RAW_PATH} and {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
