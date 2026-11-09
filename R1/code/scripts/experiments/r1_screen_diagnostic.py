"""Architecture diagnostic for candidate screening on the heterogeneous plant."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import hashlib
import json
from pathlib import Path

import r1_experiments as r1
from config import Config
from controllers import FedUMPCController


ROOT = R1_ROOT
OUT = artifact_path("results_r1_guarded")
PROTOCOL = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
SEEDS = [int(value) for value in PROTOCOL["heldout_seeds"]]


def main() -> None:
    for name, expected in PROTOCOL["frozen_source_sha256"].items():
        if hashlib.sha256((source_file(name)).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    r1.OUT = OUT
    prox = r1.load_model(artifact_path("results_r1") / "models" / "prox.pt")
    methods = {
        "No candidate screen": lambda env: [
            FedUMPCController(env, prox, use_candidate_screen=False)
            for _ in range(4)
        ],
    }
    runs, routes = r1.run_panel(
        "guarded_screen_mismatch", methods, SEEDS,
        vehicle_masses=[650.0, 850.0, 1150.0, 1350.0],
        vehicle_drags=[0.03, 0.05, 0.08, 0.11],
    )
    r1.summarize_runs(runs, "guarded_screen_mismatch")


if __name__ == "__main__":
    main()
