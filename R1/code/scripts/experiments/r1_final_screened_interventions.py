"""Run the reviewer-requested uncertainty controls under the final protocol."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import r1_experiments as r1
from config import Config
from controllers import FedUMPCController, LinearMPC


ROOT = R1_ROOT
OUT = artifact_path("results_r1_final_screened")
PROTOCOL = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
SEEDS = [int(value) for value in PROTOCOL["heldout_seeds"]]


def verify() -> None:
    for name, expected in PROTOCOL["frozen_source_sha256"].items():
        actual = hashlib.sha256((source_file(name)).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Frozen source changed: {name}")


def main() -> None:
    verify()
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    Config.RESULTS_DIR = str(OUT)
    r1.OUT = OUT
    prox = r1.load_model(artifact_path("results_r1") / "models" / "prox.pt")
    uncertainty = pd.read_csv(
        artifact_path("results_r1") / "r1_uncertainty_samples.csv"
    )["Uncertainty"].to_numpy()
    horizon = np.arange(Config.HORIZON, dtype=float)
    fixed_profile = (
        Config.SAFETY_MARGIN + Config.FED_BASE_MARGIN_OFFSET
        + np.mean(
            np.clip(
                Config.ROBUST_BETA * Config.UNCERTAINTY_MARGIN_SCALE
                * uncertainty[:, None] * (1.0 + 0.15 * horizon)[None, :],
                0.0,
                Config.MAX_UNCERTAINTY_MARGIN,
            ),
            axis=0,
        )
    )

    class FixedGrowthMPC(LinearMPC):
        def _cost(self, u_flat, state, target):
            u = u_flat.reshape(self.H, 2)
            states = self._rollout(u, state)
            return self._base_cost(states, u, target) + sum(
                self._obstacle_cost(states[index], fixed_profile[index])
                for index in range(self.H)
            )

    methods = {
        "Constant U": lambda env: [
            FedUMPCController(env, prox, uncertainty_mode="constant")
            for _ in range(4)
        ],
        "Shuffled U": lambda env: [
            FedUMPCController(env, prox, uncertainty_mode="shuffled")
            for _ in range(4)
        ],
        "Fixed-growth MPC": lambda env: [FixedGrowthMPC(env) for _ in range(4)],
    }
    runs, routes = r1.run_panel(
        "final_uncertainty_controls", methods, SEEDS
    )
    r1.summarize_runs(runs, "final_uncertainty_controls")
    r1.paired_contrasts(
        runs, routes, reference="Constant U",
        prefix="final_uncertainty_controls",
    )
    sr = runs.SR * 100.0
    runs["CBS"] = (
        2.0 * sr * runs.SC / (sr + runs.SC).clip(lower=1e-12)
        * (1.0 - runs.CR)
    )
    table = (
        runs.groupby("Method", sort=False)
        .agg(SR=("SR", "mean"), SC=("SC", "mean"),
             CR=("CR", "mean"), CBS=("CBS", "mean"))
        .reset_index()
    )
    table.SR *= 100.0
    table.CR *= 100.0
    table.to_csv(OUT / "uncertainty_controls_table.csv", index=False)
    pd.DataFrame({"h": horizon, "fixed_margin_m": fixed_profile}).to_csv(
        OUT / "fixed_growth_margin.csv", index=False
    )
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
