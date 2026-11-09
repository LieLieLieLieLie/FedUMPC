"""Freeze an independent protocol for the simplified no-adaptive-margin controller."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = R1_ROOT
OUT = artifact_path("results_r1_nomargin")
TAG = "IoT-72582-2026|FedUMPC|R1-no-adaptive-margin|2026-11-06"


def prior_seeds() -> set[int]:
    values = {11, 23, 37, 42, 71, 101, 123, 131, 456, 789, 1024,
              2048, 4096, 8192, 16384, 32768}
    for path in ROOT.glob("artifacts/**/*.csv"):
        try:
            frame = pd.read_csv(path, usecols=lambda name: name.lower() == "seed")
        except Exception:
            continue
        if len(frame.columns):
            values.update(int(value) for value in frame.iloc[:, 0].dropna())
    return values


def derive_seeds(excluded: set[int], count: int = 10) -> list[int]:
    seeds: list[int] = []
    index = 0
    while len(seeds) < count:
        digest = hashlib.sha256(f"{TAG}|{index}".encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
        index += 1
        if seed and seed not in excluded and seed not in seeds:
            seeds.append(seed)
    return seeds


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    excluded = prior_seeds()
    protocol = {
        "protocol": "Independent evaluation of simplified FedUMPC",
        "manuscript": "IoT-72582-2026",
        "design_decision": (
            "The final controller uses a fixed base clearance target. The prior "
            "adaptive margin is retained only as a controlled intervention."
        ),
        "development_and_previous_test_seeds_excluded": sorted(excluded),
        "heldout_seed_derivation_tag": TAG,
        "heldout_seed_derivation_sha256": hashlib.sha256(TAG.encode()).hexdigest(),
        "heldout_seeds": derive_seeds(excluded),
        "controller_revision": {
            "base_clearance_offset_m": 0.05,
            "adaptive_margin_in_full": False,
            "adaptive_margin_intervention_scale": 0.02,
            "uncertainty_trust_scale": 0.12,
            "near_cost_relative_tolerance": 0.002,
            "screen_horizon": 10,
            "contact_buffer_m": 0.05,
        },
        "selection_rule": (
            "Simplification follows the prior ablation diagnosis; no result from "
            "the newly derived held-out scenes is used to choose parameters."
        ),
        "scoreboard_comparator_set": [
            "Nominal MPC", "Tube-style MPC", "Scenario MPC",
            "GP-margin MPC", "Fixed-margin MPC", "FedUMPC"],
        "frozen_source_sha256": {
            name: sha256(source_file(name))
            for name in ("config.py", "controllers.py", "models.py", "main.py")
        },
    }
    path = OUT / "protocol.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != protocol:
            raise RuntimeError("Existing no-margin protocol differs from frozen content")
    else:
        path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
