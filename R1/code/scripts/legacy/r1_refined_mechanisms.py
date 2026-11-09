"""Run mechanism interventions under the frozen refined protocol."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import r1_experiments as r1
from config import Config
from controllers import FedUMPCController, LinearMPC


ROOT = R1_ROOT
OUT = artifact_path("results_r1_refined")
PROTOCOL = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
SEEDS = [int(value) for value in PROTOCOL["heldout_seeds"]]


def verify() -> None:
    for name, expected in PROTOCOL["frozen_source_sha256"].items():
        if hashlib.sha256((source_file(name)).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source changed: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["g1", "g2", "g3", "all"],
                        default="all")
    args = parser.parse_args()
    verify()
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    Config.RESULTS_DIR = str(OUT)
    r1.OUT = OUT
    prox = r1.load_model(artifact_path("results_r1") / "models" / "prox.pt")
    uncertainty = pd.read_csv(
        artifact_path("results_r1") / "r1_uncertainty_samples.csv")["Uncertainty"].to_numpy()
    h = np.arange(Config.HORIZON, dtype=float)
    fixed_profile = (Config.SAFETY_MARGIN + Config.FED_BASE_MARGIN_OFFSET
                     + np.mean(np.clip(
                         Config.ROBUST_BETA * Config.UNCERTAINTY_MARGIN_SCALE
                         * uncertainty[:, None] * (1 + 0.15*h)[None, :],
                         0.0, Config.MAX_UNCERTAINTY_MARGIN), axis=0))

    class FixedGrowthMPC(LinearMPC):
        def _cost(self, u_flat, state, target):
            u = u_flat.reshape(self.H, 2)
            states = self._rollout(u, state)
            return self._base_cost(states, u, target) + sum(
                self._obstacle_cost(states[index], fixed_profile[index])
                for index in range(self.H))

    methods = {
        "Full FedUMPC": lambda env: [FedUMPCController(env, prox)
                                      for _ in range(4)],
        "No uncertainty response": lambda env: [FedUMPCController(
            env, prox, uncertainty_aware=False) for _ in range(4)],
        "No bounded trust": lambda env: [FedUMPCController(
            env, prox, use_model_trust=False) for _ in range(4)],
        "No adaptive margin": lambda env: [FedUMPCController(
            env, prox, use_adaptive_margin=False) for _ in range(4)],
        "No candidate screen": lambda env: [FedUMPCController(
            env, prox, use_candidate_screen=False) for _ in range(4)],
        "+ uncertainty cost": lambda env: [FedUMPCController(
            env, prox, use_risk_cost=True) for _ in range(4)],
        "Constant U": lambda env: [FedUMPCController(
            env, prox, uncertainty_mode="constant") for _ in range(4)],
        "Shuffled U": lambda env: [FedUMPCController(
            env, prox, uncertainty_mode="shuffled") for _ in range(4)],
        "Fixed-growth MPC": lambda env: [FixedGrowthMPC(env) for _ in range(4)],
    }
    groups = {
        "g1": ["Full FedUMPC", "No uncertainty response", "No bounded trust"],
        "g2": ["No adaptive margin", "No candidate screen", "+ uncertainty cost"],
        "g3": ["Constant U", "Shuffled U", "Fixed-growth MPC"],
        "all": list(methods),
    }
    selected = {name: methods[name] for name in groups[args.group]}
    prefix = f"refined_mechanisms_{args.group}"
    runs, routes = r1.run_panel(prefix, selected, SEEDS)
    summary = r1.summarize_runs(runs, prefix)
    run_cbs = runs.copy()
    sr = run_cbs.SR * 100.0
    run_cbs["CBS"] = (2*sr*run_cbs.SC/(sr+run_cbs.SC).clip(lower=1e-12)
                      * (1-run_cbs.CR))
    table = (run_cbs.groupby("Method", sort=False)
             .agg(SR=("SR", "mean"), SC=("SC", "mean"),
                  CR=("CR", "mean"), CBS=("CBS", "mean"))
             .reset_index())
    table.SR *= 100.0
    table.CR *= 100.0
    table.to_csv(OUT / f"mechanism_table_{args.group}.csv", index=False)
    pd.DataFrame({"h": h, "fixed_margin_m": fixed_profile}).to_csv(
        OUT / "fixed_growth_margin.csv", index=False)
    print(summary.to_string(index=False))
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
