"""Development-only evaluation of uncertainty-aware candidate screening.

This script never writes held-out publication results.  It uses the fixed
development seeds below to compare candidate-selection variants before a
configuration is frozen for an independent evaluation set.
"""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Import the experiment runtime first: it installs the workstation's deterministic
# fallback providers before NumPy/Torch-dependent project modules are loaded.
import r1_experiments as r1
from config import Config
from controllers import FedUMPCController
from main import D_OBS, run_simulation
from utils import CrossingEnv


ROOT = R1_ROOT
MODEL_DIR = artifact_path("results_r1") / "models"
OUT = artifact_path("results_r1_dev")
OUT.mkdir(exist_ok=True)

DEVELOPMENT_SEEDS = [11, 23, 37, 71, 101, 131]


class ScreenedFedUMPC(FedUMPCController):
    """FedUMPC with a finite-candidate conservative-clearance screen.

    The screen uses only quantities already evaluated by FedUMPC.  Candidates
    that remain above their adaptive margin over the short screening horizon
    are ranked by the original objective.  If none is margin-feasible, the
    controller first avoids predicted contact and then minimizes the objective;
    if every candidate predicts contact, it maximizes the worst clearance.
    """

    def __init__(self, *args, screen_horizon=10, contact_buffer=0.10,
                 risk_mix=0.0, fallback_mode="objective", **kwargs):
        super().__init__(*args, **kwargs)
        self.screen_horizon = int(screen_horizon)
        self.contact_buffer = float(contact_buffer)
        self.risk_mix = float(risk_mix)
        if fallback_mode not in {"objective", "surplus"}:
            raise ValueError("fallback_mode must be objective or surplus")
        self.fallback_mode = fallback_mode

    def _screen_statistics(self, guesses, state):
        u_batch = np.asarray(guesses, dtype=float).reshape(-1, self.H, 2)
        hybrid, unc, physics = self._safe_bnn_rollout_batch(u_batch, state)
        margins = self._margins(unc)
        if self.risk_mix:
            prev = np.repeat(np.asarray(state, dtype=float)[None, :],
                             len(u_batch), axis=0)
            normalized_mismatch = np.empty_like(unc, dtype=float)
            for h in range(self.H):
                phy_inc = physics[:, h] - (prev if h == 0 else physics[:, h - 1])
                hyb_inc = hybrid[:, h] - (prev if h == 0 else hybrid[:, h - 1])
                normalized_mismatch[:, h] = np.linalg.norm(
                    hyb_inc - phy_inc, axis=1)
            margins = margins + np.clip(
                self.risk_mix * normalized_mismatch, 0.0, 0.8)

        clearances = np.empty((len(u_batch), self.H), dtype=float)
        for j in range(len(u_batch)):
            for h in range(self.H):
                d_phy = self.env.get_min_dist(physics[j, h, 0], physics[j, h, 1])
                d_hyb = self.env.get_min_dist(hybrid[j, h, 0], hybrid[j, h, 1])
                clearances[j, h] = min(d_phy, d_hyb)

        hs = min(self.H, self.screen_horizon)
        return (np.min(clearances[:, :hs] - margins[:, :hs], axis=1),
                np.min(clearances[:, :hs], axis=1))

    def _solve(self, state, target):
        if getattr(Config, "USE_SLSQP_REFINEMENT", False):
            return super()._solve(state, target)
        guesses = self._initial_guesses(state, target)
        values = self._cost_many(guesses, state, target)
        margin_surplus, contact_clearance = self._screen_statistics(guesses, state)

        feasible = np.flatnonzero(margin_surplus >= 0.0)
        if feasible.size:
            best_idx = int(feasible[np.nanargmin(values[feasible])])
        else:
            noncontact = np.flatnonzero(contact_clearance >= self.contact_buffer)
            if noncontact.size:
                if self.fallback_mode == "surplus":
                    best_surplus = np.nanmax(margin_surplus[noncontact])
                    near_best = noncontact[
                        margin_surplus[noncontact] >= best_surplus - 0.05]
                    best_idx = int(near_best[np.nanargmin(values[near_best])])
                else:
                    best_idx = int(noncontact[np.nanargmin(values[noncontact])])
            else:
                best_idx = int(np.nanargmax(contact_clearance))

        u_seq = np.clip(np.asarray(guesses[best_idx]), self._lo, self._hi)
        self.u_prev = np.clip(np.roll(u_seq, -2), self._lo, self._hi)
        self.u_prev[-2:] = u_seq[-2:]
        return u_seq[:2]


def evaluate() -> pd.DataFrame:
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.USE_SLSQP_REFINEMENT = False
    prox = r1.load_model(MODEL_DIR / "prox.pt")
    fedavg = r1.load_model(MODEL_DIR / "fedavg.pt")
    configs = {
        "prox_original": (prox, None),
        "fedavg_original": (fedavg, None),
        "prox_screen_h6": (prox, dict(screen_horizon=6, contact_buffer=0.05)),
        "prox_screen_h10": (prox, dict(screen_horizon=10, contact_buffer=0.05)),
        "prox_screen_h6_surplus": (prox, dict(
            screen_horizon=6, contact_buffer=0.05, fallback_mode="surplus")),
        "prox_screen_h10_surplus": (prox, dict(
            screen_horizon=10, contact_buffer=0.05, fallback_mode="surplus")),
    }
    run_path = OUT / "selector_development_runs.csv"
    if run_path.exists():
        rows = pd.read_csv(run_path).to_dict("records")
    else:
        rows = []
    completed = {(str(row["Configuration"]), int(row["Seed"]))
                 for row in rows}
    for name, (model, screening) in configs.items():
        for seed in DEVELOPMENT_SEEDS:
            if (name, int(seed)) in completed:
                continue
            print(f"[development] {name} seed={seed}", flush=True)
            r1.reset_rng(seed)
            env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
            if screening is None:
                controllers = [FedUMPCController(env, model) for _ in range(4)]
            else:
                controllers = [ScreenedFedUMPC(env, model, **screening)
                               for _ in range(4)]
            result = run_simulation(controllers, env, seed=seed, strict=True)
            rows.append({
                "Configuration": name,
                "Seed": seed,
                "SR_pct": 100 * result["SR"],
                "SC_pct": result["SC"],
                "CR_pct": 100 * result["CR"],
                "MinClearance_m": result["MinClearance"],
                "MeanViolationDepth_m": result["MeanViolationDepth"],
                "ComputeMs": result["ComputeMs"],
            })
            pd.DataFrame(rows).to_csv(run_path, index=False)
            completed.add((name, int(seed)))
    frame = pd.DataFrame(rows)
    summary = (frame.groupby("Configuration", sort=False)
               .agg({"SR_pct": ["mean", "std"],
                     "SC_pct": ["mean", "std"],
                     "CR_pct": ["mean", "std"],
                     "MinClearance_m": ["mean", "std"],
                     "MeanViolationDepth_m": ["mean", "std"],
                     "ComputeMs": ["mean", "std"]}))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / "selector_development_summary.csv",
                                 index=False)
    (OUT / "selector_development_protocol.json").write_text(json.dumps({
        "purpose": "development-only algorithm selection",
        "development_seeds": DEVELOPMENT_SEEDS,
        "heldout_results_used_for_selection": False,
        "candidate_variants": list(configs),
    }, indent=2), encoding="utf-8")
    print(summary.to_string())
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    evaluate()
