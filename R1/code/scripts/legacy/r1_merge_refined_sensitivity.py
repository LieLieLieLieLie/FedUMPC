"""Merge independently checkpointed obstacle-potential sensitivity parts."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from pathlib import Path

import pandas as pd


def main():
    out = artifact_path("results_r1_refined")
    frames = [pd.read_csv(out / "sensitivity_parts" / f"{name}.csv")
              for name in ["q_obs", "c_s", "inner_a", "d_inf"]]
    frame = pd.concat(frames, ignore_index=True)
    frame.to_csv(out / "r1_obstacle_sensitivity.csv", index=False)
    (frame.groupby(["Parameter", "Base", "Value"], as_index=False)
          .agg(SR=("SR", "mean"), SC=("SC", "mean"),
               MinClearance=("MinClearance", "mean"),
               MeanViolationDepth=("MeanViolationDepth", "mean"),
               Cost=("Cost", "mean"))
          .to_csv(out / "r1_obstacle_sensitivity_summary.csv", index=False))


if __name__ == "__main__":
    main()
