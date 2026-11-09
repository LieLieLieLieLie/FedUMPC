"""Run only mechanism interventions not covered by the completed diagnostic panel."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from pathlib import Path

import pandas as pd

import r1_experiments as r1
from config import Config
from controllers import FedUMPCController


ROOT = R1_ROOT
OUT = artifact_path("results_r1_mechanisms_final")
SEEDS = [338953336, 888537210, 296369628, 919042070, 1902629867,
         986992663, 301465597, 654430970, 1405222590, 470027004]


def main() -> None:
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
    methods = {
        "No bounded trust": lambda env: [FedUMPCController(
            env, prox, use_model_trust=False) for _ in range(4)],
        "No candidate screen": lambda env: [FedUMPCController(
            env, prox, use_candidate_screen=False) for _ in range(4)],
        "Constant U": lambda env: [FedUMPCController(
            env, prox, uncertainty_mode="constant") for _ in range(4)],
        "Shuffled U": lambda env: [FedUMPCController(
            env, prox, uncertainty_mode="shuffled") for _ in range(4)],
    }
    runs, routes = r1.run_panel("final_mechanism_missing", methods, SEEDS)
    r1.summarize_runs(runs, "final_mechanism_missing")
    # Preserve the raw routes for paired merging with the completed panel.
    pd.DataFrame(runs).to_csv(OUT / "final_mechanism_missing_runs.csv", index=False)
    pd.DataFrame(routes).to_csv(OUT / "final_mechanism_missing_routes.csv", index=False)


if __name__ == "__main__":
    main()
