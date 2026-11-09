"""Checkpoint worker for one frozen robustness condition.

The parent script remains the only process that assembles the final tables.
This worker merely fills the same per-method/per-seed checkpoints so that
independent disturbance conditions can be evaluated concurrently.
"""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import argparse

import r1_experiments as r1
import r1_final_robustness as robustness
from config import Config

final = robustness.final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("condition", choices=list(robustness.CONDITIONS))
    args = parser.parse_args()
    final.verify()
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    Config.SEEDS = final.SEEDS
    Config.N_SEEDS = len(final.SEEDS)
    Config.RESULTS_DIR = str(robustness.OUT)
    r1.OUT = robustness.OUT
    prox = r1.load_model(robustness.artifact_path("results_r1") / "models" / "prox.pt")
    all_methods = robustness.factories(prox)
    methods = {name: all_methods[name] for name in robustness.METHODS}
    values = robustness.CONDITIONS[args.condition]
    slug = args.condition.lower().replace(" ", "_").replace(".", "p")
    r1.run_panel(f"final_robustness_{slug}", methods, final.SEEDS,
                 noise=values["noise"], disturbance=values["disturbance"])


if __name__ == "__main__":
    main()
