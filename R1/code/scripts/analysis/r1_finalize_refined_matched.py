"""Merge matched predictor comparisons with the refined FedUMPC reference."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from pathlib import Path

import pandas as pd


ROOT = R1_ROOT
OUT = artifact_path("results_r1_refined")


def main() -> None:
    runs = []
    prox = pd.read_csv(OUT / "refined_mismatch_runs.csv")
    runs.append(prox[prox.Method.eq("FedUMPC")])
    for key in ("local", "fedavg", "ensemble", "central"):
        runs.append(pd.read_csv(OUT / f"refined_matched_{key}_runs.csv"))
    frame = pd.concat(runs, ignore_index=True)
    sr = frame.SR * 100.0
    frame["CBS"] = 2*sr*frame.SC/(sr+frame.SC).clip(lower=1e-12)*(1-frame.CR)
    summary = (frame.groupby("Method", sort=False)
               .agg(SR=("SR", "mean"), SC=("SC", "mean"),
                    CR=("CR", "mean"), CBS=("CBS", "mean"),
                    MinClearance=("MinClearance", "mean"))
               .reset_index())
    summary.SR *= 100.0
    summary.CR *= 100.0
    frame.to_csv(OUT / "refined_matched_runs.csv", index=False)
    summary.to_csv(OUT / "refined_matched_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
