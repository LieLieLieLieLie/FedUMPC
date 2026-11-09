"""Run common- and heterogeneous-plant comparisons for guarded FedUMPC."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import hashlib
import json
from pathlib import Path

import pandas as pd

import r1_experiments as r1
from config import Config
from controllers import (FedUMPCController, GP_MPC, LinearMPC, RobustMPC,
                         StochasticMPC, TubeMPC)


ROOT = R1_ROOT
OUT = artifact_path("results_r1_guarded")
PROTOCOL = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
SEEDS = [int(value) for value in PROTOCOL["heldout_seeds"]]


def verify() -> None:
    for name, expected in PROTOCOL["frozen_source_sha256"].items():
        if hashlib.sha256((source_file(name)).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source changed: {name}")


def cbs(frame: pd.DataFrame) -> pd.Series:
    sr = frame.SR * 100.0
    return 2 * sr * frame.SC / (sr + frame.SC).clip(lower=1e-12) * (1-frame.CR)


def main() -> None:
    verify()
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    Config.RESULTS_DIR = str(OUT)
    r1.OUT = OUT
    prox = r1.load_model(artifact_path("results_r1") / "models" / "prox.pt")
    methods = {
        "Nominal MPC": lambda env: [LinearMPC(env) for _ in range(4)],
        "Tube-style MPC": lambda env: [TubeMPC(env) for _ in range(4)],
        "Scenario MPC": lambda env: [StochasticMPC(env) for _ in range(4)],
        "GP-margin MPC": lambda env: [GP_MPC(env) for _ in range(4)],
        "Fixed-margin MPC": lambda env: [RobustMPC(env) for _ in range(4)],
        "FedUMPC": lambda env: [FedUMPCController(env, prox) for _ in range(4)],
    }
    scoreboards = []
    for regime, kwargs, prefix in [
        ("Common", {}, "guarded_common"),
        ("Heterogeneous", {
            "vehicle_masses": [650.0, 850.0, 1150.0, 1350.0],
            "vehicle_drags": [0.03, 0.05, 0.08, 0.11],
        }, "guarded_mismatch"),
    ]:
        runs, routes = r1.run_panel(prefix, methods, SEEDS, **kwargs)
        r1.summarize_runs(runs, prefix)
        r1.paired_contrasts(runs, routes, reference="FedUMPC", prefix=prefix)
        selected = runs[runs.Method.isin(PROTOCOL["scoreboard_comparator_set"])].copy()
        selected["CBS"] = cbs(selected)
        summary = (selected.groupby("Method", sort=False)
                   .agg(SR=("SR", "mean"), SC=("SC", "mean"),
                        CR=("CR", "mean"), CBS=("CBS", "mean"),
                        MinClearance=("MinClearance", "mean"),
                        MeanViolationDepth=("MeanViolationDepth", "mean"),
                        MaxViolationDuration=("MaxViolationDuration", "mean"))
                   .reset_index())
        summary.SR *= 100.0
        summary.CR *= 100.0
        summary.insert(0, "Regime", regime)
        scoreboards.append(summary)
    pd.concat(scoreboards, ignore_index=True).to_csv(
        OUT / "seven_metric_scoreboard.csv", index=False)


if __name__ == "__main__":
    main()
