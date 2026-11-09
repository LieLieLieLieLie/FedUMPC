"""Frozen disturbance-robustness evaluation for the R1 revision.

The controller sources and held-out seeds are inherited from the independent
evaluation protocol.  Conditions are declared below before any outcome is
computed; the script checkpoints every method--seed pair through run_panel.
"""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import hashlib
import json
from pathlib import Path

import pandas as pd

import r1_experiments as r1
import r1_frozen_evaluation as frozen
from config import Config


ROOT = R1_ROOT
OUT = artifact_path("results_r1_frozen")
CONDITIONS = {
    "Nominal": {"noise": 0.00, "disturbance": 0.00},
    "Process noise 0.10": {"noise": 0.10, "disturbance": 0.00},
    "Process noise 0.25": {"noise": 0.25, "disturbance": 0.00},
    "Actuation bias 0.50": {"noise": 0.00, "disturbance": 0.50},
    "Compound stress": {"noise": 0.25, "disturbance": 1.00},
}
METHOD_ORDER = [
    "Nominal MPC", "Tube-style MPC", "Scenario MPC", "GP-margin MPC",
    "Fixed-margin MPC", "FedUMPC",
]


def write_protocol() -> None:
    path = OUT / "robustness_protocol.json"
    parent_bytes = (OUT / "protocol.json").read_bytes()
    payload = {
        "parent_protocol_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "heldout_seeds": frozen.SEEDS,
        "conditions": CONDITIONS,
        "methods": METHOD_ORDER,
        "metrics": ["SR", "SC", "CR", "MinClearance",
                    "MeanViolationDepth", "MaxViolationDuration"],
    }
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != payload:
            raise RuntimeError("The frozen robustness protocol has changed")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    frozen.verify_frozen_sources()
    write_protocol()
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    Config.SEEDS = frozen.SEEDS
    Config.N_SEEDS = len(frozen.SEEDS)
    Config.RESULTS_DIR = str(OUT)
    r1.OUT = OUT

    prox, fedavg, central, local, ensemble = frozen.models()
    all_methods = frozen.factories(prox, fedavg, central, local, ensemble)
    methods = {name: all_methods[name] for name in METHOD_ORDER}
    run_frames = []
    route_frames = []
    for condition, values in CONDITIONS.items():
        slug = condition.lower().replace(" ", "_").replace(".", "p")
        runs, routes = r1.run_panel(
            f"frozen_robustness_{slug}", methods, frozen.SEEDS,
            noise=values["noise"], disturbance=values["disturbance"])
        runs.insert(0, "Condition", condition)
        routes.insert(0, "Condition", condition)
        run_frames.append(runs)
        route_frames.append(routes)

    runs = pd.concat(run_frames, ignore_index=True)
    routes = pd.concat(route_frames, ignore_index=True)
    runs.to_csv(OUT / "frozen_robustness_runs.csv", index=False)
    routes.to_csv(OUT / "frozen_robustness_routes.csv", index=False)

    summary = (runs.groupby(["Condition", "Method"], sort=False)
               .agg(SR_mean=("SR", "mean"), SR_std=("SR", "std"),
                    SC_mean=("SC", "mean"), SC_std=("SC", "std"),
                    CR_mean=("CR", "mean"), CR_std=("CR", "std"),
                    MinClearance_mean=("MinClearance", "mean"),
                    MinClearance_std=("MinClearance", "std"),
                    MeanViolationDepth_mean=("MeanViolationDepth", "mean"),
                    MeanViolationDepth_std=("MeanViolationDepth", "std"),
                    MaxViolationDuration_mean=("MaxViolationDuration", "mean"),
                    MaxViolationDuration_std=("MaxViolationDuration", "std"))
               .reset_index())
    for metric in ("SR_mean", "SC_mean", "CR_mean"):
        summary[metric] *= 100.0
    summary.to_csv(OUT / "frozen_robustness_summary.csv", index=False)
    print(summary[["Condition", "Method", "SR_mean", "SC_mean",
                   "CR_mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
