"""Merge nominal and disturbed refined-controller evidence."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from pathlib import Path

import pandas as pd


ROOT = R1_ROOT
OUT = artifact_path("results_r1_refined")
SOURCES = {
    "Nominal": "refined_common",
    "Process noise 0.10": "refined_robustness_noise010",
    "Process noise 0.25": "refined_robustness_noise025",
    "Actuation bias 0.50": "refined_robustness_bias050",
    "Compound stress": "refined_robustness_compound",
}


def main() -> None:
    run_frames, route_frames = [], []
    for condition, prefix in SOURCES.items():
        runs = pd.read_csv(OUT / f"{prefix}_runs.csv")
        routes = pd.read_csv(OUT / f"{prefix}_routes.csv")
        runs.insert(0, "Condition", condition)
        routes.insert(0, "Condition", condition)
        run_frames.append(runs); route_frames.append(routes)
    runs = pd.concat(run_frames, ignore_index=True)
    routes = pd.concat(route_frames, ignore_index=True)
    runs.to_csv(OUT / "refined_robustness_runs.csv", index=False)
    routes.to_csv(OUT / "refined_robustness_routes.csv", index=False)
    summary = (runs.groupby(["Condition", "Method"], sort=False)
               .agg(SR_mean=("SR", "mean"), SR_std=("SR", "std"),
                    SC_mean=("SC", "mean"), SC_std=("SC", "std"),
                    CR_mean=("CR", "mean"), CR_std=("CR", "std"),
                    MinClearance_mean=("MinClearance", "mean"),
                    MinClearance_std=("MinClearance", "std"),
                    MeanViolationDepth_mean=("MeanViolationDepth", "mean"),
                    MeanViolationDepth_std=("MeanViolationDepth", "std"),
                    MaxViolationDuration_mean=("MaxViolationDuration", "mean"),
                    MaxViolationDuration_std=("MaxViolationDuration", "std"))
               .reset_index())
    summary[["SR_mean", "SR_std", "CR_mean", "CR_std"]] *= 100.0
    summary.to_csv(OUT / "refined_robustness_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
