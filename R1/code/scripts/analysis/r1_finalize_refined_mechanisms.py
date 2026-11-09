"""Merge the three disjoint refined-mechanism checkpoints."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from pathlib import Path

import pandas as pd

import r1_experiments as r1


ROOT = R1_ROOT
OUT = artifact_path("results_r1_refined")


def main() -> None:
    runs = pd.concat([
        pd.read_csv(OUT / f"refined_mechanisms_{group}_runs.csv")
        for group in ("g1", "g2", "g3")], ignore_index=True)
    routes = pd.concat([
        pd.read_csv(OUT / f"refined_mechanisms_{group}_routes.csv")
        for group in ("g1", "g2", "g3")], ignore_index=True)
    runs.to_csv(OUT / "refined_mechanisms_runs.csv", index=False)
    routes.to_csv(OUT / "refined_mechanisms_routes.csv", index=False)
    r1.OUT = OUT
    r1.summarize_runs(runs, "refined_mechanisms")
    r1.paired_contrasts(runs, routes, reference="Full FedUMPC",
                        prefix="refined_mechanisms")
    frame = runs.copy()
    sr = frame.SR * 100.0
    frame["CBS"] = 2*sr*frame.SC/(sr+frame.SC).clip(lower=1e-12)*(1-frame.CR)
    table = (frame.groupby("Method", sort=False)
             .agg(SR=("SR", "mean"), SC=("SC", "mean"),
                  CR=("CR", "mean"), CBS=("CBS", "mean"))
             .reset_index())
    table.SR *= 100.0
    table.CR *= 100.0
    table.to_csv(OUT / "mechanism_table.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
