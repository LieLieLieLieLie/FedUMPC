"""Run the final common- and heterogeneous-plant comparison."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import json
from pathlib import Path

import r1_guarded_evaluation as implementation


ROOT = R1_ROOT
implementation.OUT = artifact_path("results_r1_final_screened")
implementation.PROTOCOL = json.loads(
    (implementation.OUT / "protocol.json").read_text(encoding="utf-8"))
implementation.SEEDS = [
    int(value) for value in implementation.PROTOCOL["heldout_seeds"]]


if __name__ == "__main__":
    implementation.main()
