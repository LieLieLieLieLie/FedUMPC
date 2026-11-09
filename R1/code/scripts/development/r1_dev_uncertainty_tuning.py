"""Development-only calibration of uncertainty-to-trust sensitivity.

Each scale is evaluated with estimated, constant, and deterministically
shuffled uncertainty on the same six development scenes.  Selection rewards
closed-loop endpoints and the decision value of candidate-specific estimates;
publication holdout records are never read.
"""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import json
from pathlib import Path

import numpy as np
import pandas as pd

import r1_experiments as r1
from config import Config
from main import D_OBS, run_simulation
from r1_dev_screen_tuning import DevelopmentController, cbs
from utils import CrossingEnv


ROOT = R1_ROOT
OUT = artifact_path("results_r1_dev_uncertainty")
MODEL = artifact_path("results_r1") / "models" / "prox.pt"
SEEDS = [11, 23, 37, 71, 101, 131]
# The endpoints are compared at the existing scale and at one deliberately
# more uncertainty-sensitive endpoint.  Keeping this two-point structural
# comparison avoids a broad hyperparameter search on a small development set.
SCALES = [0.12, 0.55]
MODES = ["estimated", "constant", "shuffled"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "runs.csv"
    rows = pd.read_csv(path).to_dict("records") if path.exists() else []
    complete = {(float(row["Scale"]), str(row["Mode"]), int(row["Seed"]))
                for row in rows}
    model = r1.load_model(MODEL)
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    for scale in SCALES:
        Config.BNN_TRUST_UNC_SCALE = scale
        for mode in MODES:
            for seed in SEEDS:
                key = (scale, mode, seed)
                if key in complete:
                    continue
                print(f"[uncertainty development] scale={scale:.2f} "
                      f"mode={mode} seed={seed}", flush=True)
                r1.reset_rng(seed)
                env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
                controllers = [DevelopmentController(
                    env, model, uncertainty_mode=mode, screen_horizon=10,
                    contact_buffer=0.05, screen_offset=0.05,
                    selection_slack=0.002) for _ in range(4)]
                result = run_simulation(controllers, env, seed=seed, strict=True)
                rows.append({
                    "Scale": scale, "Mode": mode, "Seed": seed,
                    "SR": 100.0 * result["SR"], "SC": result["SC"],
                    "CR": 100.0 * result["CR"],
                    "MinClearance": result["MinClearance"],
                    "MeanViolationDepth": result["MeanViolationDepth"],
                    "MaxViolationDuration": result["MaxViolationDuration"],
                })
                pd.DataFrame(rows).to_csv(path, index=False)
                complete.add(key)

    frame = pd.DataFrame(rows)
    summary = (frame.groupby(["Scale", "Mode"], sort=False)
               .agg(SR=("SR", "mean"), SC=("SC", "mean"),
                    CR=("CR", "mean"),
                    MinClearance=("MinClearance", "mean"),
                    MeanViolationDepth=("MeanViolationDepth", "mean"),
                    MaxViolationDuration=("MaxViolationDuration", "mean"))
               .reset_index())
    summary["CBS"] = [cbs(a, b, c) for a, b, c in
                      zip(summary.SR, summary.SC, summary.CR)]
    summary.to_csv(OUT / "summary.csv", index=False)

    candidates = []
    for scale in SCALES:
        block = summary[np.isclose(summary.Scale, scale)].set_index("Mode")
        full = block.loc["estimated"]
        best_count = 0
        signed_gaps = []
        for metric, direction in {"SR": "max", "SC": "max", "CR": "min",
                                  "CBS": "max"}.items():
            values = block[metric]
            target = getattr(values, direction)()
            best_count += int(np.isclose(full[metric], target))
            alternatives = values.drop("estimated")
            gap = (full[metric] - alternatives.max() if direction == "max"
                   else alternatives.min() - full[metric])
            signed_gaps.append(float(gap))
        candidates.append({
            "Scale": scale, "FullBestCount": best_count,
            "FullMeanSignedGap": float(np.mean(signed_gaps)),
            "FullCBS": float(full.CBS), "FullSC": float(full.SC),
            "FullSR": float(full.SR), "FullCR": float(full.CR),
        })
    ranking = pd.DataFrame(candidates).sort_values(
        ["FullBestCount", "FullCBS", "FullMeanSignedGap", "FullSC"],
        ascending=[False, False, False, False])
    ranking.to_csv(OUT / "scale_ranking.csv", index=False)
    payload = {
        "development_seeds": SEEDS,
        "heldout_results_used_for_selection": False,
        "fixed_screen_offset_m": 0.05,
        "fixed_relative_near_cost_tolerance": 0.002,
        "selection_rule": (
            "maximize estimated-uncertainty first-place count over SR/SC/CR/CBS, "
            "then its CBS, mean signed gap, and SC"),
        "selected": ranking.iloc[0].to_dict(),
    }
    (OUT / "selection.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
