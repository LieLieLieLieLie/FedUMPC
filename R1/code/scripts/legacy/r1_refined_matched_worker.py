"""Evaluate one matched predictor under the refined heterogeneous controller."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import argparse
import json
from pathlib import Path

import r1_experiments as r1
from config import Config
from controllers import FedUMPCController
from models import EnsemblePredictor


ROOT = R1_ROOT
OUT = artifact_path("results_r1_refined")
PROTOCOL = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
SEEDS = [int(value) for value in PROTOCOL["heldout_seeds"]]
MODEL_DIR = artifact_path("results_r1") / "models"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["local", "fedavg", "ensemble", "central"])
    args = parser.parse_args()
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    r1.OUT = OUT
    if args.method == "local":
        models = [r1.load_model(MODEL_DIR / f"local_{index}.pt") for index in range(4)]
        label = "Local UMPC"
        factory = lambda env: [FedUMPCController(env, models[index])
                               for index in range(4)]
    elif args.method == "fedavg":
        model = r1.load_model(MODEL_DIR / "fedavg.pt")
        label = "FedAvg UMPC"
        factory = lambda env: [FedUMPCController(env, model) for _ in range(4)]
    elif args.method == "central":
        model = r1.load_model(MODEL_DIR / "central.pt")
        label = "Centralized UMPC"
        factory = lambda env: [FedUMPCController(env, model) for _ in range(4)]
    else:
        model = EnsemblePredictor([
            r1.load_model(MODEL_DIR / f"ensemble_{index}.pt") for index in range(3)])
        label = "Ensemble UMPC"
        factory = lambda env: [FedUMPCController(env, model) for _ in range(4)]
    r1.run_panel(
        f"refined_matched_{args.method}", {label: factory}, SEEDS,
        vehicle_masses=[650.0, 850.0, 1150.0, 1350.0],
        vehicle_drags=[0.03, 0.05, 0.08, 0.11])


if __name__ == "__main__":
    main()
