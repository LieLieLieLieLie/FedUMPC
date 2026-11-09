"""Evaluate one frozen mechanism intervention in an isolated checkpoint."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import argparse

import r1_experiments as r1
from config import Config
from controllers import FedUMPCController
from r1_final_mechanism_interventions import OUT, SEEDS, ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["constant", "shuffled"])
    args = parser.parse_args()
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    Config.SEEDS = SEEDS
    Config.N_SEEDS = len(SEEDS)
    Config.RESULTS_DIR = str(OUT)
    OUT.mkdir(parents=True, exist_ok=True)
    r1.OUT = OUT
    prox = r1.load_model(artifact_path("results_r1") / "models" / "prox.pt")
    label = "Constant U" if args.mode == "constant" else "Shuffled U"
    methods = {label: lambda env: [FedUMPCController(
        env, prox, uncertainty_mode=args.mode) for _ in range(4)]}
    r1.run_panel(f"final_mechanism_{args.mode}", methods, SEEDS)


if __name__ == "__main__":
    main()
