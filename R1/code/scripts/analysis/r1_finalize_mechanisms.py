"""Assemble the final mechanism table from disjoint, completed checkpoints."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from pathlib import Path

import pandas as pd


ROOT = R1_ROOT
OUT = artifact_path("results_r1_mechanisms_final")


def add_cbs(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    sr = frame["SR"] * 100.0
    frame["CBS"] = 2.0 * sr * frame["SC"] / (sr + frame["SC"]) * (1.0 - frame["CR"])
    return frame


def main() -> None:
    diagnostic = pd.read_csv(artifact_path("results_r1_frozen") / "r1_mechanisms_runs.csv")
    diagnostic = diagnostic[diagnostic["Method"].isin([
        "Trust only", "Risk + trust", "Trust + margin", "Full FedUMPC",
        "Fixed-growth MPC",
    ])]
    diagnostic["Method"] = diagnostic["Method"].replace({
        "Trust only": "FedUMPC",
        "Risk + trust": "+ direct uncertainty cost",
        "Trust + margin": "+ adaptive margin",
        "Full FedUMPC": "+ cost and margin",
    })
    missing = pd.read_csv(OUT / "final_mechanism_missing_runs.csv")
    constant = pd.read_csv(OUT / "final_mechanism_constant_runs.csv")
    shuffled = pd.read_csv(OUT / "final_mechanism_shuffled_runs.csv")
    runs = pd.concat([diagnostic, missing, constant, shuffled], ignore_index=True)
    runs = runs.drop_duplicates(subset=["Method", "Seed"], keep="last")
    runs = add_cbs(runs)
    runs.to_csv(OUT / "final_mechanism_runs.csv", index=False)
    summary = (runs.groupby("Method", sort=False)
               .agg(SR_mean=("SR", "mean"), SC_mean=("SC", "mean"),
                    CR_mean=("CR", "mean"), CBS_mean=("CBS", "mean"),
                    MinClearance_mean=("MinClearance", "mean"),
                    MeanViolationDepth_mean=("MeanViolationDepth", "mean"),
                    MaxViolationDuration_mean=("MaxViolationDuration", "mean"))
               .reset_index())
    summary[["SR_mean", "CR_mean"]] *= 100.0
    summary.to_csv(OUT / "final_mechanism_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
