"""Development-only matched-predictor control comparison."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from pathlib import Path

import pandas as pd

import r1_experiments as r1
from controllers import FedUMPCController
from main import D_OBS, run_simulation
from models import EnsemblePredictor
from r1_dev_selector import DEVELOPMENT_SEEDS
from utils import CrossingEnv


ROOT = R1_ROOT
OUT = artifact_path("results_r1_dev")
PATH = OUT / "learner_development_runs.csv"
MODEL_DIR = artifact_path("results_r1") / "models"


def main():
    models = {
        "Prox-FL": r1.load_model(MODEL_DIR / "prox.pt"),
        "FedAvg": r1.load_model(MODEL_DIR / "fedavg.pt"),
        "Centralized": r1.load_model(MODEL_DIR / "central.pt"),
        "Ensemble": EnsemblePredictor([
            r1.load_model(MODEL_DIR / f"ensemble_{i}.pt") for i in range(3)]),
    }
    locals_ = [r1.load_model(MODEL_DIR / f"local_{i}.pt") for i in range(4)]
    rows = pd.read_csv(PATH).to_dict("records") if PATH.exists() else []
    completed = {(str(row["Method"]), int(row["Seed"])) for row in rows}
    for method in ["Prox-FL", "FedAvg", "Centralized", "Ensemble", "Local"]:
        for seed in DEVELOPMENT_SEEDS:
            if (method, seed) in completed:
                continue
            print(f"[development] {method} screen-h10 seed={seed}", flush=True)
            r1.reset_rng(seed)
            env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
            predictors = locals_ if method == "Local" else [models[method]] * 4
            result = run_simulation(
                [FedUMPCController(env, predictor) for predictor in predictors],
                env, seed=seed, strict=True)
            rows.append({
                "Method": method, "Seed": seed,
                "SR_pct": 100 * result["SR"], "SC_pct": result["SC"],
                "CR_pct": 100 * result["CR"],
                "MinClearance_m": result["MinClearance"],
                "MeanViolationDepth_m": result["MeanViolationDepth"],
                "ComputeMs": result["ComputeMs"],
            })
            pd.DataFrame(rows).to_csv(PATH, index=False)
            completed.add((method, seed))
    frame = pd.DataFrame(rows)
    summary = (frame.groupby("Method", sort=False)
               .agg({"SR_pct": ["mean", "std"], "SC_pct": ["mean", "std"],
                     "CR_pct": ["mean", "std"],
                     "MinClearance_m": ["mean", "std"],
                     "MeanViolationDepth_m": ["mean", "std"],
                     "ComputeMs": ["mean", "std"]}))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / "learner_development_summary.csv",
                                 index=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
