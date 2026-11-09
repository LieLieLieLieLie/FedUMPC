"""Execute the frozen independent R1 control evaluation.

The held-out seeds and source hashes are read from ``results_r1_frozen``.
Any change to the four frozen algorithm files aborts the run, preventing
post-outcome controller tuning from silently entering the reported benchmark.
"""

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
from models import EnsemblePredictor


ROOT = R1_ROOT
OUT = artifact_path("results_r1_frozen")
MODEL_DIR = artifact_path("results_r1") / "models"
PROTOCOL = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
SEEDS = [int(seed) for seed in PROTOCOL["heldout_seeds"]]


def verify_frozen_sources():
    for name, expected in PROTOCOL["frozen_source_sha256"].items():
        actual = hashlib.sha256((source_file(name)).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(
                f"Frozen source changed: {name}; start a new protocol before evaluation")


def models():
    prox = r1.load_model(MODEL_DIR / "prox.pt")
    fedavg = r1.load_model(MODEL_DIR / "fedavg.pt")
    central = r1.load_model(MODEL_DIR / "central.pt")
    local = [r1.load_model(MODEL_DIR / f"local_{i}.pt") for i in range(4)]
    ensemble = EnsemblePredictor([
        r1.load_model(MODEL_DIR / f"ensemble_{i}.pt") for i in range(3)])
    return prox, fedavg, central, local, ensemble


def factories(prox, fedavg, central, local, ensemble):
    return {
        "Nominal MPC": lambda env: [LinearMPC(env) for _ in range(4)],
        "Tube-style MPC": lambda env: [TubeMPC(env) for _ in range(4)],
        "Scenario MPC": lambda env: [StochasticMPC(env) for _ in range(4)],
        "GP-margin MPC": lambda env: [GP_MPC(env) for _ in range(4)],
        "Fixed-margin MPC": lambda env: [RobustMPC(env) for _ in range(4)],
        "Local UMPC": lambda env: [FedUMPCController(env, local[i])
                                     for i in range(4)],
        "FedAvg UMPC": lambda env: [FedUMPCController(env, fedavg)
                                      for _ in range(4)],
        "Ensemble UMPC": lambda env: [FedUMPCController(env, ensemble)
                                        for _ in range(4)],
        "Centralized UMPC": lambda env: [FedUMPCController(env, central)
                                           for _ in range(4)],
        "FedUMPC": lambda env: [FedUMPCController(env, prox)
                                  for _ in range(4)],
    }


def add_composite(summary_path: Path):
    frame = pd.read_csv(summary_path)
    sr = frame["SR_mean"]
    sc = frame["SC_mean"]
    cr = frame["CR_mean"]
    frame["CBS_mean"] = (2.0 * sr * sc / (sr + sc).clip(lower=1e-12)
                         * (1.0 - cr / 100.0))
    frame.to_csv(summary_path, index=False)


def main():
    verify_frozen_sources()
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    Config.SEEDS = SEEDS
    Config.N_SEEDS = len(SEEDS)
    r1.OUT = OUT
    Config.RESULTS_DIR = str(OUT)

    prox, fedavg, central, local, ensemble = models()
    methods = factories(prox, fedavg, central, local, ensemble)

    common_runs, common_routes = r1.run_panel(
        "frozen_common", methods, SEEDS)
    r1.summarize_runs(common_runs, "frozen_common")
    r1.paired_contrasts(common_runs, common_routes,
                        reference="FedUMPC", prefix="frozen_common")
    add_composite(OUT / "frozen_common_summary.csv")

    mismatch_runs, mismatch_routes = r1.run_panel(
        "frozen_mismatch", methods, SEEDS,
        vehicle_masses=[650.0, 850.0, 1150.0, 1350.0],
        vehicle_drags=[0.03, 0.05, 0.08, 0.11])
    r1.summarize_runs(mismatch_runs, "frozen_mismatch")
    r1.paired_contrasts(mismatch_runs, mismatch_routes,
                        reference="FedUMPC", prefix="frozen_mismatch")
    add_composite(OUT / "frozen_mismatch_summary.csv")
    print(pd.read_csv(OUT / "frozen_common_summary.csv")[
        ["Method", "SR_mean", "SC_mean", "CR_mean", "CBS_mean"]]
        .to_string(index=False))
    print(pd.read_csv(OUT / "frozen_mismatch_summary.csv")[
        ["Method", "SR_mean", "SC_mean", "CR_mean", "CBS_mean"]]
        .to_string(index=False))


if __name__ == "__main__":
    main()
