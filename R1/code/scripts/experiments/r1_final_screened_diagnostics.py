"""Recompute deployment diagnostics for the final screened controller."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import json
from pathlib import Path

import r1_experiments as exp
from config import Config


def main():
    root = R1_ROOT
    output = artifact_path("results_r1_final_screened")
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    exp.OUT = output
    Config.RESULTS_DIR = str(output)
    exp.MC_TEST_SEEDS = [int(value) for value in protocol["heldout_seeds"]]
    exp.SENSITIVITY_SEEDS = [11, 23, 37, 71, 101, 131]
    Config.set_mode("fast")
    _, metadata, prox, *_ = exp.prepare_primary_models()
    exp.mc_sample_sweep(prox, metadata)
    exp.candidate_scaling(prox)
    exp.obstacle_sensitivity(prox)


if __name__ == "__main__":
    main()
