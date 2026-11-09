"""Independent multi-scene MuJoCo evaluation after the R1 screen was frozen."""

from __future__ import annotations

import csv
import hashlib
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
    _fallback_rng = py_random.Random(20261104)

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
OUTPUT = ROOT / "outputs_r1_frozen"
PROTOCOL_PATH = OUTPUT / "protocol.json"
RAW_PATH = OUTPUT / "mujoco_runs.csv"
SUMMARY_PATH = OUTPUT / "mujoco_summary.csv"
TAG = "IoT-72582-2026|FedUMPC|R1-MuJoCo-independent|screen-h10|2026-11-04"
EXCLUDED = {11, 23, 42, 123, 456, 789, 1024}


def derive_seeds(count=5):
    seeds = []
    counter = 0
    while len(seeds) < count:
        digest = hashlib.sha256(f"{TAG}|{counter}".encode()).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        if seed not in EXCLUDED and seed not in seeds:
            seeds.append(seed)
        counter += 1
    return seeds


SEEDS = derive_seeds()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_payload():
    return {
        "protocol": "Independent MuJoCo evaluation after method freeze",
        "seed_derivation_tag": TAG,
        "seed_derivation_sha256": hashlib.sha256(TAG.encode()).hexdigest(),
        "heldout_seeds": SEEDS,
        "excluded_development_and_previous_seeds": sorted(EXCLUDED),
        "methods": list(core.METHODS),
        "screen_horizon_steps": core.FED_SCREEN_HORIZON,
        "predicted_noncontact_buffer_m": core.FED_CONTACT_BUFFER,
        "source_sha256": {
            "run_six_algorithms.py": sha256(ROOT / "run_six_algorithms.py"),
            "r1_mujoco_frozen.py": sha256(Path(__file__).resolve()),
        },
    }


def freeze_or_verify_protocol():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = protocol_payload()
    if PROTOCOL_PATH.exists():
        recorded = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
        if recorded != payload:
            raise RuntimeError("Frozen MuJoCo protocol or source changed")
    else:
        PROTOCOL_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def existing_keys():
    if not RAW_PATH.exists():
        return set()
    with RAW_PATH.open(newline="", encoding="utf-8-sig") as stream:
        return {(int(row["seed"]), row["method"])
                for row in csv.DictReader(stream)}


def append_row(row):
    write_header = not RAW_PATH.exists()
    with RAW_PATH.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def write_summary():
    with RAW_PATH.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    numeric = [
        "success", "collisions", "mean_clearance", "min_clearance",
        "mean_steps", "violation_frequency_pct", "violation_episodes",
        "mean_violation_depth", "max_violation_depth",
        "max_violation_duration_s",
    ]
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)
    summary = []
    for method in core.METHODS:
        values = grouped.get(method, [])
        if not values:
            continue
        result = {"method": method, "scenes": len(values)}
        for name in numeric:
            array = np.asarray([float(row[name]) for row in values])
            result[f"{name}_mean"] = float(array.mean())
            result[f"{name}_sample_sd"] = (
                float(array.std(ddof=1)) if len(array) > 1 else 0.0)
        summary.append(result)
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def main():
    freeze_or_verify_protocol()
    done = existing_keys()
    for seed in SEEDS:
        obstacles = core.generate_obstacles(seed)
        xml = core.make_scene_xml(obstacles)
        model = core.NumpyDropoutBNN(seed=seed)
        print(f"[frozen MuJoCo] training predictor | seed={seed}", flush=True)
        model.fit_federated(seed=seed)
        for method in core.METHODS:
            if (seed, method) in done:
                continue
            if method == "FedUMPC":
                model.rng = np.random.default_rng(seed + 9000)
            print(f"[frozen MuJoCo] {method} | seed={seed}", flush=True)
            _, _, metrics = core.run_method(
                method, xml, obstacles, model, show_viewer=False)
            append_row({"seed": seed, **asdict(metrics)})
            write_summary()
    print(f"Wrote {RAW_PATH} and {SUMMARY_PATH}", flush=True)


if __name__ == "__main__":
    main()
