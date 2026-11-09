"""Development-only selection of a nominal-anchored acceptance gate.

The guard compares the uncertainty-aware candidate with the candidate selected
by the nominal objective from the same generated set.  It accepts the former
only when its nominal cost remains within a fixed tolerance and its predicted
conservative clearance is no smaller.  All choices are made on the six fixed
development scenes; publication holdout results are never read here.
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
OUT = artifact_path("results_r1_dev_nominal_guard")
MODEL = artifact_path("results_r1") / "models" / "prox.pt"
SEEDS = [11, 23, 37, 71, 101, 131]
GRID = [
    ("gain000_slack000", 0.00, 0.000),
    ("gain000_slack002", 0.00, 0.002),
    ("gain002_slack002", 0.02, 0.002),
    ("gain005_slack010", 0.05, 0.010),
]


class GuardedFedUMPC(FedUMPCController):
    def __init__(self, *args, clearance_gain=0.0, nominal_slack=0.0, **kwargs):
        super().__init__(*args, use_adaptive_margin=False, **kwargs)
        self.clearance_gain = float(clearance_gain)
        self.nominal_slack = float(nominal_slack)

    def _nominal_scores(self, guesses, state, target):
        scores = []
        for guess in guesses:
            u = np.asarray(guess, dtype=float).reshape(self.H, 2)
            states = self._rollout(u, state)
            score = self._base_cost(states, u, target)
            score += sum(self._obstacle_cost(s, Config.SAFETY_MARGIN)
                         for s in states)
            scores.append(score)
        return np.asarray(scores, dtype=float)

    def _solve(self, state, target):
        guesses = self._initial_guesses(state, target)
        values, bnn_states, unc, phy_states = self._evaluate_many(
            guesses, state, target)
        admissible, noncontact, worst_surplus, worst_clearance = (
            self._screen_candidates(bnn_states, unc, phy_states))
        if admissible.size:
            fed_idx = self._choose_near_optimal(
                admissible, values, worst_clearance)
        elif noncontact.size:
            fed_idx = self._choose_near_optimal(
                noncontact, values, worst_clearance)
        else:
            fed_idx = int(np.nanargmax(worst_clearance))

        nominal_values = self._nominal_scores(guesses, state, target)
        nominal_idx = int(np.nanargmin(nominal_values))
        tolerance = self.nominal_slack * max(
            abs(float(nominal_values[nominal_idx])), 1.0)
        acceptable_cost = (
            nominal_values[fed_idx] <= nominal_values[nominal_idx] + tolerance)
        acceptable_clearance = (
            worst_clearance[fed_idx] >=
            worst_clearance[nominal_idx] + self.clearance_gain)
        nominal_predicted_contact = (
            worst_clearance[nominal_idx] < self.contact_buffer)
        fed_predicted_admissible = fed_idx in set(admissible.tolist())
        if ((acceptable_cost and acceptable_clearance) or
                (nominal_predicted_contact and fed_predicted_admissible)):
            best_idx = fed_idx
        else:
            best_idx = nominal_idx
        sequence = np.clip(np.asarray(guesses[best_idx]), self._lo, self._hi)
        self.u_prev = np.clip(np.roll(sequence, -2), self._lo, self._hi)
        self.u_prev[-2:] = sequence[-2:]
        return sequence[:2]


def cbs(sr, sc, cr):
    return 2.0 * sr * sc / max(sr + sc, 1e-12) * (1.0 - cr / 100.0)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    complete = {(str(row["Config"]), int(row["Seed"])) for row in rows}
    model = r1.load_model(MODEL)
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    methods = [(name, gain, slack) for name, gain, slack in GRID]
    methods.append(("No uncertainty response", None, None))
    for name, gain, slack in methods:
        for seed in SEEDS:
            if (name, seed) in complete:
                continue
            print(f"[nominal guard development] {name} seed={seed}", flush=True)
            r1.reset_rng(seed)
            env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
            if name == "No uncertainty response":
                controllers = [FedUMPCController(
                    env, model, uncertainty_aware=False) for _ in range(4)]
            else:
                controllers = [GuardedFedUMPC(
                    env, model, clearance_gain=gain, nominal_slack=slack)
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
               .agg(SR=("SR", "mean"), SC=("SC", "mean"), CR=("CR", "mean"),
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
    candidates = summary[summary.Config != "No uncertainty response"].copy()
    candidates = candidates.sort_values(
        ["PrimaryBestCount", "CBS", "SC", "SR", "CR"],
        ascending=[False, False, False, False, True])
    selected = candidates.iloc[0].to_dict()
    summary.to_csv(OUT / "summary.csv", index=False)
    (OUT / "selection.json").write_text(json.dumps({
        "development_seeds": SEEDS,
        "publication_holdout_read": False,
        "selection_rule": (
            "maximize first-place count over SR/SC/CR/CBS against all guard "
            "configurations and the no-uncertainty control, then CBS and SC"),
        "grid": GRID,
        "selected": selected,
    }, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
