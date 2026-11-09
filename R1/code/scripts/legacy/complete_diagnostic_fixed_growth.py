"""Complete the four interrupted fixed-growth diagnostic scenes only."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from pathlib import Path

import numpy as np
import pandas as pd

import r1_experiments as r1
from config import Config
from controllers import LinearMPC


ROOT = R1_ROOT
OUT = artifact_path("results_r1_frozen")
ALL_SEEDS = [338953336, 888537210, 296369628, 919042070, 1902629867,
             986992663, 301465597, 654430970, 1405222590, 470027004]


def main() -> None:
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    r1.OUT = OUT
    uncertainty = pd.read_csv(
        artifact_path("results_r1") / "r1_uncertainty_samples.csv")["Uncertainty"].to_numpy()
    calibration_u = uncertainty.reshape(-1, 1)
    fixed_profile = Config.SAFETY_MARGIN + np.mean(np.clip(
        Config.ROBUST_BETA * Config.UNCERTAINTY_MARGIN_SCALE
        * calibration_u * (1 + 0.15 * np.arange(Config.HORIZON))[None, :],
        0.0, Config.MAX_UNCERTAINTY_MARGIN), axis=0)

    class FixedGrowthMPC(LinearMPC):
        def _cost(self, u_flat, state, target):
            u = u_flat.reshape(self.H, 2)
            states = self._rollout(u, state)
            return self._base_cost(states, u, target) + sum(
                self._obstacle_cost(states[h], fixed_profile[h])
                for h in range(self.H))

    existing = pd.read_csv(OUT / "r1_mechanisms_runs.csv")
    completed = set(existing.loc[
        existing.Method.eq("Fixed-growth MPC"), "Seed"].astype(int))
    missing = [seed for seed in ALL_SEEDS if seed not in completed]
    if missing:
        runs, routes = r1.run_panel(
            "diagnostic_fixed_growth_missing",
            {"Fixed-growth MPC": lambda env: [FixedGrowthMPC(env)
                                                for _ in range(4)]},
            missing)
        old_routes = pd.read_csv(OUT / "r1_mechanisms_routes.csv")
        merged_runs = pd.concat([existing, runs], ignore_index=True)
        merged_routes = pd.concat([old_routes, routes], ignore_index=True)
        merged_runs.to_csv(OUT / "r1_mechanisms_runs.csv", index=False)
        merged_routes.to_csv(OUT / "r1_mechanisms_routes.csv", index=False)
    final_runs = pd.read_csv(OUT / "r1_mechanisms_runs.csv")
    final_routes = pd.read_csv(OUT / "r1_mechanisms_routes.csv")
    r1.summarize_runs(final_runs, "r1_mechanisms")
    r1.paired_contrasts(final_runs, final_routes,
                        reference="Risk + trust", prefix="r1_mechanisms")


if __name__ == "__main__":
    main()
