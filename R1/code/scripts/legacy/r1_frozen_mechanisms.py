"""Run the predeclared mechanism-isolation panel on held-out R1 seeds."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import hashlib
import json
from pathlib import Path

import pandas as pd

import r1_experiments as r1
import r1_frozen_evaluation as frozen
from config import Config


ROOT = R1_ROOT
OUT = artifact_path("results_r1_frozen")
METHODS = [
    "Full FedUMPC", "No uncertainty response", "Risk only", "Trust only",
    "Margin only", "Risk + trust", "Risk + margin", "Trust + margin",
    "No candidate screen", "Constant U", "Shuffled U", "Fixed-growth MPC",
]


def write_protocol() -> None:
    path = OUT / "mechanism_protocol.json"
    parent_bytes = (OUT / "protocol.json").read_bytes()
    payload = {
        "parent_protocol_sha256": hashlib.sha256(parent_bytes).hexdigest(),
        "heldout_seeds": frozen.SEEDS,
        "methods": METHODS,
        "fixed_growth_calibration": (
            "mean clipped horizon-wise adaptive margin on the independent "
            "held-out uncertainty-query set"),
        "primary_endpoints": frozen.PROTOCOL["primary_endpoints"],
    }
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise RuntimeError("The frozen mechanism protocol has changed")
    else:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    frozen.verify_frozen_sources()
    write_protocol()
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    Config.SEEDS = frozen.SEEDS
    Config.N_SEEDS = len(frozen.SEEDS)
    Config.RESULTS_DIR = str(OUT)
    r1.OUT = OUT
    r1.TEST_SEEDS = list(frozen.SEEDS)

    prox, _, _, _, _ = frozen.models()
    uncertainty_path = artifact_path("results_r1") / "r1_uncertainty_samples.csv"
    uncertainty = pd.read_csv(uncertainty_path)["Uncertainty"].to_numpy()
    summary = r1.mechanism_isolation(prox, uncertainty)
    observed = set(summary["Method"])
    if observed != set(METHODS):
        raise RuntimeError(f"Mechanism set mismatch: {sorted(observed)}")
    print(summary[["Method", "SR_mean", "SC_mean", "CR_mean"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
