"""Recompute deployment diagnostics for the frozen R1 controller.

The primary predictor is loaded from the existing matched-training cache.  The
controller parameters are already frozen in ``config.py``; these sweeps are
descriptive and do not feed back into parameter selection.
"""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from pathlib import Path

import r1_experiments as exp
from config import Config


def main():
    refined = artifact_path("results_r1_refined")
    refined.mkdir(parents=True, exist_ok=True)
    exp.OUT = refined
    Config.RESULTS_DIR = str(refined)

    # Ten frozen formal scenes for the MC-sample control diagnostic; six
    # disjoint development scenes for local obstacle-potential sensitivity.
    exp.MC_TEST_SEEDS = [
        1609894196, 1519566093, 712774354, 1220388233, 925882782,
        989563491, 1352891092, 380348096, 1183841035, 1396745301,
    ]
    exp.SENSITIVITY_SEEDS = [11, 23, 37, 71, 101, 131]

    Config.set_mode("fast")
    _, metadata, prox, *_ = exp.prepare_primary_models()
    exp.mc_sample_sweep(prox, metadata)
    exp.candidate_scaling(prox)
    exp.obstacle_sensitivity(prox)


if __name__ == "__main__":
    main()
