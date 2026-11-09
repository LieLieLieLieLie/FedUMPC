"""Development-only baseline targets for controller selection."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from pathlib import Path

import pandas as pd

import r1_experiments as r1
from controllers import GP_MPC, LinearMPC, RobustMPC, StochasticMPC, TubeMPC
from main import D_OBS, run_simulation
from r1_dev_selector import DEVELOPMENT_SEEDS
from utils import CrossingEnv


OUT = artifact_path("results_r1_dev")
PATH = OUT / "baseline_development_runs.csv"


def main():
    methods = {
        "Nominal MPC": LinearMPC,
        "Tube-style MPC": TubeMPC,
        "Scenario MPC": StochasticMPC,
        "GP-margin MPC": GP_MPC,
        "Fixed-margin MPC": RobustMPC,
    }
    rows = pd.read_csv(PATH).to_dict("records") if PATH.exists() else []
    completed = {(str(row["Method"]), int(row["Seed"])) for row in rows}
    for method, controller_class in methods.items():
        for seed in DEVELOPMENT_SEEDS:
            if (method, seed) in completed:
                continue
            print(f"[development] {method} seed={seed}", flush=True)
            r1.reset_rng(seed)
            env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
            result = run_simulation([controller_class(env) for _ in range(4)],
                                    env, seed=seed, strict=True)
            rows.append({
                "Method": method,
                "Seed": seed,
                "SR_pct": 100 * result["SR"],
                "SC_pct": result["SC"],
                "CR_pct": 100 * result["CR"],
                "MinClearance_m": result["MinClearance"],
                "MeanViolationDepth_m": result["MeanViolationDepth"],
                "ComputeMs": result["ComputeMs"],
            })
            pd.DataFrame(rows).to_csv(PATH, index=False)
            completed.add((method, seed))
    frame = pd.DataFrame(rows)
    summary = (frame.groupby("Method", sort=False)
               .agg({"SR_pct": ["mean", "std"],
                     "SC_pct": ["mean", "std"],
                     "CR_pct": ["mean", "std"],
                     "MinClearance_m": ["mean", "std"],
                     "MeanViolationDepth_m": ["mean", "std"],
                     "ComputeMs": ["mean", "std"]}))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / "baseline_development_summary.csv",
                                 index=False)
    print(summary.to_string())


if __name__ == "__main__":
    main()
