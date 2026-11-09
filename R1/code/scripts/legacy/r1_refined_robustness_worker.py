"""Run one refined-controller disturbance condition into an isolated checkpoint."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import argparse
import hashlib
import json
from pathlib import Path

import r1_experiments as r1
from config import Config
from controllers import (FedUMPCController, GP_MPC, LinearMPC, RobustMPC,
                         StochasticMPC, TubeMPC)


ROOT = R1_ROOT
OUT = artifact_path("results_r1_refined")
PROTOCOL = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
SEEDS = [int(value) for value in PROTOCOL["heldout_seeds"]]
CONDITIONS = {
    "noise010": {"noise": 0.10, "disturbance": 0.00},
    "noise025": {"noise": 0.25, "disturbance": 0.00},
    "bias050": {"noise": 0.00, "disturbance": 0.50},
    "compound": {"noise": 0.25, "disturbance": 1.00},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=CONDITIONS)
    args = parser.parse_args()
    for name, expected in PROTOCOL["frozen_source_sha256"].items():
        if hashlib.sha256((source_file(name)).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
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
    values = CONDITIONS[args.condition]
    r1.run_panel(f"refined_robustness_{args.condition}", methods, SEEDS,
                 noise=values["noise"], disturbance=values["disturbance"])


if __name__ == "__main__":
    main()
