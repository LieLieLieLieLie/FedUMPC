"""Development-only audit for the implementation-matched MuJoCo adapter."""

from __future__ import annotations

import csv
import inspect
import os
import random as py_random
import sys
import types
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
OUTPUT = ROOT / "outputs_r1_adapter_audit"
SEEDS = (11, 23)
TRAINING_SEED = 20261104


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictor = core.NumpyDropoutBNN(seed=TRAINING_SEED)
    predictor.fit_federated(seed=TRAINING_SEED, actual_plant=True)
    rows = []
    for seed in SEEDS:
        obstacles = core.generate_obstacles(seed)
        xml = core.make_scene_xml(obstacles)
        for method in core.METHODS:
            if method == "FedUMPC":
                predictor.rng = np.random.default_rng(seed + 9000)
            print(f"[adapter audit] {method} | seed={seed}", flush=True)
            _, _, metrics = core.run_method(
                method, xml, obstacles, predictor, show_viewer=False)
            rows.append({"seed": seed, **asdict(metrics)})
    destination = OUTPUT / "development_runs.csv"
    with destination.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
