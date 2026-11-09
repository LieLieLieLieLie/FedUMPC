"""Parallel worker for the frozen-controller obstacle-potential sweep."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import sys
from pathlib import Path

import numpy as np

import r1_experiments as exp
from config import Config
from controllers import FedUMPCController
from main import D_OBS, run_simulation
from utils import CrossingEnv


SETTINGS = {
    "q_obs": (7800.0, [5460.0, 7800.0, 10140.0]),
    "c_s": (0.02, [0.01, 0.02, 0.04]),
    "inner_a": (1.15, [0.80, 1.15, 1.50]),
    "d_inf": (4.2, [3.8, 4.2, 4.8]),
}
SEEDS = [11, 23, 37, 71, 101, 131]


def main(parameter):
    if parameter not in SETTINGS:
        raise SystemExit(f"unknown parameter: {parameter}")
    root = R1_ROOT
    part_dir = artifact_path("results_r1_refined") / "sensitivity_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    path = part_dir / f"{parameter}.csv"

    Config.set_mode("fast")
    _, _, prox, *_ = exp.prepare_primary_models()
    base, values = SETTINGS[parameter]
    done = exp.completed_keys(path, ["Parameter", "Value", "Seed"])
    for value in values:
        for seed in SEEDS:
            if (parameter, value, seed) in done:
                continue
            print(f"[{parameter}] value={value} seed={seed}", flush=True)
            updates = {
                "OBSTACLE_PENALTY": value if parameter == "q_obs" else 7800.0,
                "FED_SOFT_OBS_WEIGHT": value if parameter == "c_s" else 0.02,
                "FED_INNER_OBS_WEIGHT": value if parameter == "inner_a" else 1.15,
                "FED_OBSTACLE_INFLUENCE": value if parameter == "d_inf" else 4.2,
            }
            with exp.temporary_config(**updates):
                env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
                result = run_simulation(
                    [FedUMPCController(env, prox) for _ in range(4)], env,
                    seed=seed)
            scalar = {key: val for key, val in result.items()
                      if isinstance(val, (int, float, np.number))}
            exp.append_row(path, {"Parameter": parameter, "Base": base,
                                  "Value": value, "Seed": seed, **scalar})


if __name__ == "__main__":
    main(sys.argv[1])
