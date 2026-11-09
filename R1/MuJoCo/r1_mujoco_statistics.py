"""Paired scene-level statistics for the frozen R1 MuJoCo evaluation."""

from __future__ import annotations

import csv
import os
import random as py_random
from collections import defaultdict
from pathlib import Path

try:
    os.urandom(1)
except OSError:
    _fallback_rng = py_random.Random(20261105)

    def _deterministic_urandom(n):
        return bytes(_fallback_rng.getrandbits(8) for _ in range(n))

    os.urandom = _deterministic_urandom
    py_random._urandom = _deterministic_urandom

import numpy as np


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs_r1_final"
RAW_PATH = OUTPUT / "mujoco_runs.csv"
CONTRAST_PATH = OUTPUT / "mujoco_paired_contrasts.csv"
RANK_PATH = OUTPUT / "mujoco_endpoint_first_place.csv"
FED = "FedUMPC"
BOOTSTRAPS = 10_000

METRICS = {
    "success_rate_pct": ("success", 25.0, "higher"),
    "contact_rate_pct": ("collisions", 25.0, "lower"),
    "min_clearance_m": ("min_clearance", 1.0, "higher"),
    "violation_frequency_pct": ("violation_frequency_pct", 1.0, "lower"),
    "mean_violation_depth_m": ("mean_violation_depth", 1.0, "lower"),
    "max_violation_duration_s": ("max_violation_duration_s", 1.0, "lower"),
}


def load_rows():
    with RAW_PATH.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def paired_bootstrap(difference, rng):
    n = difference.size
    indices = rng.integers(0, n, size=(BOOTSTRAPS, n))
    sampled = difference[indices].mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def main():
    rows = load_rows()
    by_method = defaultdict(dict)
    for row in rows:
        by_method[row["method"]][int(row["seed"])] = row
    seeds = sorted(by_method[FED])
    if any(sorted(values) != seeds for values in by_method.values()):
        raise RuntimeError("Methods do not share the same ten held-out scene seeds")

    rng = np.random.default_rng(20261105)
    contrasts = []
    for comparator in by_method:
        if comparator == FED:
            continue
        for label, (column, scale, direction) in METRICS.items():
            fed = np.asarray([float(by_method[FED][seed][column]) * scale
                              for seed in seeds])
            other = np.asarray([float(by_method[comparator][seed][column]) * scale
                                for seed in seeds])
            difference = fed - other
            low, high = paired_bootstrap(difference, rng)
            contrasts.append({
                "metric": label,
                "direction": direction,
                "comparator": comparator,
                "fed_mean": float(fed.mean()),
                "comparator_mean": float(other.mean()),
                "fed_minus_comparator": float(difference.mean()),
                "ci95_low": low,
                "ci95_high": high,
            })
    with CONTRAST_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(contrasts[0]))
        writer.writeheader()
        writer.writerows(contrasts)

    ranks = []
    for label, (column, scale, direction) in METRICS.items():
        means = {
            method: np.mean([float(by_method[method][seed][column]) * scale
                             for seed in seeds])
            for method in by_method
        }
        best = (max(means.values()) if direction == "higher"
                else min(means.values()))
        for method, mean in means.items():
            ranks.append({
                "metric": label,
                "direction": direction,
                "method": method,
                "mean": float(mean),
                "first_place": int(np.isclose(mean, best, rtol=0.0, atol=1e-12)),
            })
    with RANK_PATH.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranks[0]))
        writer.writeheader()
        writer.writerows(ranks)
    print(f"Wrote {CONTRAST_PATH} and {RANK_PATH}")


if __name__ == "__main__":
    main()
