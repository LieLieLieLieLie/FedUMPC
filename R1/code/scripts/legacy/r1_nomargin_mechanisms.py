"""Run core mechanism interventions for the simplified FedUMPC controller."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import hashlib
import json
from pathlib import Path

import pandas as pd

import r1_experiments as r1
from config import Config
from controllers import FedUMPCController


ROOT = R1_ROOT
OUT = artifact_path("results_r1_nomargin")
PROTOCOL = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
SEEDS = [int(value) for value in PROTOCOL["heldout_seeds"]]


def verify() -> None:
    for name, expected in PROTOCOL["frozen_source_sha256"].items():
        if hashlib.sha256((source_file(name)).read_bytes()).hexdigest() != expected:
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
    methods = {
        "Full FedUMPC": lambda env: [FedUMPCController(
            env, prox, use_adaptive_margin=False) for _ in range(4)],
        "No uncertainty response": lambda env: [FedUMPCController(
            env, prox, uncertainty_aware=False) for _ in range(4)],
        "No bounded trust": lambda env: [FedUMPCController(
            env, prox, use_model_trust=False, use_adaptive_margin=False)
            for _ in range(4)],
        "No candidate screen": lambda env: [FedUMPCController(
            env, prox, use_candidate_screen=False, use_adaptive_margin=False)
            for _ in range(4)],
        "+ adaptive margin": lambda env: [FedUMPCController(
            env, prox, use_adaptive_margin=True) for _ in range(4)],
        "+ uncertainty cost": lambda env: [FedUMPCController(
            env, prox, use_risk_cost=True, use_adaptive_margin=False)
            for _ in range(4)],
    }
    runs, routes = r1.run_panel("nomargin_mechanisms", methods, SEEDS)
    r1.summarize_runs(runs, "nomargin_mechanisms")
    sr = runs.SR * 100.0
    runs["CBS"] = (2*sr*runs.SC/(sr+runs.SC).clip(lower=1e-12)
                   * (1-runs.CR))
    table = (runs.groupby("Method", sort=False)
             .agg(SR=("SR", "mean"), SC=("SC", "mean"),
                  CR=("CR", "mean"), CBS=("CBS", "mean"))
             .reset_index())
    table.SR *= 100.0
    table.CR *= 100.0
    table.to_csv(OUT / "mechanism_table.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
