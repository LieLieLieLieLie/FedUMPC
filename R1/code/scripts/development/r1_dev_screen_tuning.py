"""Development-only tuning of the finite candidate screen.

The grid is evaluated only on the six fixed development seeds used before the
independent R1 evaluation.  It changes no metric and never reads publication
holdout records.  A fresh protocol must be frozen after selection.
"""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import json
from pathlib import Path

import numpy as np
import pandas as pd

import r1_experiments as r1
from config import Config
from controllers import FedUMPCController
from main import D_OBS, run_simulation
from utils import CrossingEnv


ROOT = R1_ROOT
OUT = artifact_path("results_r1_dev_screen_v3")
MODEL = artifact_path("results_r1") / "models" / "prox.pt"
SEEDS = [11, 23, 37, 71, 101, 131]

# name, screen horizon, fixed screening offset, relative near-optimal cost slack
GRID = [
    ("h10_o000_s000", 10, 0.00, 0.000),
    ("h10_o005_s002", 10, 0.05, 0.002),
    ("h10_o010_s002", 10, 0.10, 0.002),
    ("h10_o015_s002", 10, 0.15, 0.002),
]


class DevelopmentController(FedUMPCController):
    def __init__(self, *args, screen_offset=0.0, selection_slack=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.screen_offset = float(screen_offset)
        self.selection_slack = float(selection_slack)

    def _margins(self, uncertainty):
        """Apply the development offset consistently in cost and screening."""
        return super()._margins(uncertainty) + self.screen_offset

    def _screen_candidates(self, bnn_states, unc, phy_states):
        margins = self._margins(unc)
        clearances = np.empty((len(phy_states), self.H), dtype=float)
        for j in range(len(phy_states)):
            for h in range(self.H):
                d_phy = self.env.get_min_dist(
                    phy_states[j, h, 0], phy_states[j, h, 1])
                d_hyb = self.env.get_min_dist(
                    bnn_states[j, h, 0], bnn_states[j, h, 1])
                clearances[j, h] = min(d_phy, d_hyb)
        hs = min(self.H, max(1, self.screen_horizon))
        worst_surplus = np.min(clearances[:, :hs] - margins[:, :hs], axis=1)
        worst_clearance = np.min(clearances[:, :hs], axis=1)
        admissible = np.flatnonzero(worst_surplus >= 0.0)
        noncontact = np.flatnonzero(worst_clearance >= self.contact_buffer)
        return admissible, noncontact, worst_surplus, worst_clearance

    def _choose_near_optimal(self, eligible, values, clearance):
        minimum = float(np.nanmin(values[eligible]))
        tolerance = self.selection_slack * max(abs(minimum), 1.0)
        near = eligible[values[eligible] <= minimum + tolerance]
        if len(near) == 1 or self.selection_slack <= 0.0:
            return int(eligible[np.nanargmin(values[eligible])])
        safest = near[np.isclose(clearance[near], np.nanmax(clearance[near]))]
        return int(safest[np.nanargmin(values[safest])])

    def _solve(self, state, target):
        guesses = self._initial_guesses(state, target)
        values, bnn_states, unc, phy_states = self._evaluate_many(
            guesses, state, target)
        admissible, noncontact, _, worst_clearance = self._screen_candidates(
            bnn_states, unc, phy_states)
        if admissible.size:
            best = self._choose_near_optimal(
                admissible, values, worst_clearance)
        elif noncontact.size:
            best = self._choose_near_optimal(
                noncontact, values, worst_clearance)
        else:
            best = int(np.nanargmax(worst_clearance))
        sequence = np.clip(np.asarray(guesses[best]), self._lo, self._hi)
        self.u_prev = np.clip(np.roll(sequence, -2), self._lo, self._hi)
        self.u_prev[-2:] = sequence[-2:]
        return sequence[:2]


def cbs(sr, sc, cr):
    return 2.0 * sr * sc / max(sr + sc, 1e-12) * (1.0 - cr / 100.0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    complete = {(str(row["Config"]), int(row["Seed"])) for row in rows}
    model = r1.load_model(MODEL)
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    for name, horizon, offset, slack in GRID:
        for seed in SEEDS:
            if (name, seed) in complete:
                continue
            print(f"[screen development] {name} seed={seed}", flush=True)
            r1.reset_rng(seed)
            env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
            controllers = [DevelopmentController(
                env, model, screen_horizon=horizon, contact_buffer=0.05,
                screen_offset=offset, selection_slack=slack)
                for _ in range(4)]
            result = run_simulation(controllers, env, seed=seed, strict=True)
            rows.append({
                "Config": name, "Seed": seed,
                "SR": 100.0 * result["SR"], "SC": result["SC"],
                "CR": 100.0 * result["CR"],
                "MinClearance": result["MinClearance"],
                "MeanViolationDepth": result["MeanViolationDepth"],
                "MaxViolationDuration": result["MaxViolationDuration"],
            })
            pd.DataFrame(rows).to_csv(path, index=False)
            complete.add((name, seed))
    frame = pd.DataFrame(rows)
    summary = (frame.groupby("Config", sort=False)
               .agg(SR=("SR", "mean"), SC=("SC", "mean"),
                    CR=("CR", "mean"),
                    MinClearance=("MinClearance", "mean"),
                    MeanViolationDepth=("MeanViolationDepth", "mean"),
                    MaxViolationDuration=("MaxViolationDuration", "mean"))
               .reset_index())
    summary["CBS"] = [cbs(a, b, c) for a, b, c in
                      zip(summary.SR, summary.SC, summary.CR)]
    summary["PrimaryBestCount"] = 0
    for metric, direction in {"SR": "max", "SC": "max", "CR": "min",
                              "CBS": "max"}.items():
        best = getattr(summary[metric], direction)()
        summary["PrimaryBestCount"] += np.isclose(summary[metric], best)
    summary = summary.sort_values(
        ["PrimaryBestCount", "CBS", "SC", "SR", "CR"],
        ascending=[False, False, False, False, True])
    summary.to_csv(OUT / "summary.csv", index=False)
    selected = summary.iloc[0].to_dict()
    (OUT / "selection.json").write_text(json.dumps({
        "development_seeds": SEEDS,
        "heldout_results_used_for_selection": False,
        "selection_rule": (
            "maximize first-place count over SR/SC/CR/CBS, then CBS, SC, SR, CR"),
        "grid": GRID,
        "selected": selected,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
