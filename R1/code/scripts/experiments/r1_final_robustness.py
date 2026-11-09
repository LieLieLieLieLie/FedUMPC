"""Run the frozen final controller across predeclared disturbance regimes."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import json
from pathlib import Path

import pandas as pd

import r1_experiments as r1
import r1_guarded_evaluation as final
from config import Config
from controllers import (FedUMPCController, GP_MPC, LinearMPC, RobustMPC,
                         StochasticMPC, TubeMPC)


ROOT = R1_ROOT
OUT = artifact_path("results_r1_final_screened")
final.OUT = OUT
final.PROTOCOL = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
final.SEEDS = [int(value) for value in final.PROTOCOL["heldout_seeds"]]
CONDITIONS = {
    "Nominal": {"noise": 0.00, "disturbance": 0.00},
    "Process noise 0.10": {"noise": 0.10, "disturbance": 0.00},
    "Process noise 0.25": {"noise": 0.25, "disturbance": 0.00},
    "Actuation bias 0.50": {"noise": 0.00, "disturbance": 0.50},
    "Compound stress": {"noise": 0.25, "disturbance": 1.00},
}
METHODS = list(final.PROTOCOL["scoreboard_comparator_set"])


def factories(prox):
    return {
        "Nominal MPC": lambda env: [LinearMPC(env) for _ in range(4)],
        "Tube-style MPC": lambda env: [TubeMPC(env) for _ in range(4)],
        "Scenario MPC": lambda env: [StochasticMPC(env) for _ in range(4)],
        "GP-margin MPC": lambda env: [GP_MPC(env) for _ in range(4)],
        "Fixed-margin MPC": lambda env: [RobustMPC(env) for _ in range(4)],
        "FedUMPC": lambda env: [FedUMPCController(env, prox) for _ in range(4)],
    }


def main() -> None:
    final.verify()
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    Config.SEEDS = final.SEEDS
    Config.N_SEEDS = len(final.SEEDS)
    Config.RESULTS_DIR = str(OUT)
    r1.OUT = OUT
    prox = r1.load_model(artifact_path("results_r1") / "models" / "prox.pt")
    all_methods = factories(prox)
    methods = {name: all_methods[name] for name in METHODS}
    protocol = {
        "parent_protocol": json.loads((OUT / "protocol.json").read_text(encoding="utf-8")),
        "conditions": CONDITIONS, "methods": METHODS,
    }
    path = OUT / "robustness_protocol.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Robustness protocol changed")
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    run_frames, route_frames = [], []
    for condition, values in CONDITIONS.items():
        slug = condition.lower().replace(" ", "_").replace(".", "p")
        runs, routes = r1.run_panel(
            f"final_robustness_{slug}", methods, final.SEEDS,
            noise=values["noise"], disturbance=values["disturbance"])
        runs.insert(0, "Condition", condition)
        routes.insert(0, "Condition", condition)
        run_frames.append(runs); route_frames.append(routes)
    runs = pd.concat(run_frames, ignore_index=True)
    routes = pd.concat(route_frames, ignore_index=True)
    # Checkpoints may be filled by independent condition workers.  A worker
    # interrupted near a completed write can leave a repeated record; retain
    # one deterministic observation per experimental unit before aggregation.
    runs = runs.drop_duplicates(
        subset=["Condition", "Method", "Seed"], keep="last")
    routes = routes.drop_duplicates(
        subset=["Condition", "Method", "Seed", "vehicle"], keep="last")
    runs.to_csv(OUT / "final_robustness_runs.csv", index=False)
    routes.to_csv(OUT / "final_robustness_routes.csv", index=False)
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
    summary[["SR_mean", "SR_std", "CR_mean", "CR_std"]] *= 100.0
    summary.to_csv(OUT / "final_robustness_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
