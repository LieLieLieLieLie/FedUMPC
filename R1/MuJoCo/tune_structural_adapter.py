"""Tune clearance-aware candidate tie-breaking on development evidence only."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import r1_mujoco_final as compatibility
import run_six_algorithms as core


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "development_structural_tuning"
EARLY_SEEDS = [11, 23, 37]
DIAGNOSTIC_PROTOCOL = json.loads(
    (ROOT / "diagnostic_r1d_outputs" / "protocol.json").read_text(encoding="utf-8"))
DIAGNOSTIC_SEEDS = [int(value) for value in DIAGNOSTIC_PROTOCOL["heldout_seeds"]]
SEEDS = EARLY_SEEDS + DIAGNOSTIC_SEEDS
# The trust cap and soft-obstacle weight were frozen by the preceding
# development-only adapter study.  This second-stage study changes only the
# clearance-aware tie-breaking tolerance, avoiding a joint retune after the
# earlier diagnostic evidence was inspected.
CAPS = [0.05]
SOFT_WEIGHTS = [150.0]
COST_SLACKS = [0.0005, 0.0010, 0.0020]
METRICS = {
    "SR": "max", "SC": "max", "CR": "min", "CBS": "max",
    "MinClearance": "max", "MeanViolationDepth": "min",
    "MaxViolationDuration": "min",
}


def load(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def append(path: Path, row: dict) -> None:
    header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if header:
            writer.writeheader()
        writer.writerow(row)


def aggregate(items: list[dict]) -> dict[str, float]:
    success = np.mean([float(row["success"]) for row in items]) / 4 * 100
    contact = np.mean([float(row["collisions"]) for row in items]) / 4 * 100
    sc = 100 - np.mean([float(row["violation_frequency_pct"]) for row in items])
    return {
        "SR": success, "SC": sc, "CR": contact,
        "CBS": 2 * success * sc / max(success + sc, 1e-12)
               * (1 - contact / 100),
        "MinClearance": np.mean([float(row["min_clearance"]) for row in items]),
        "MeanViolationDepth": np.mean([
            float(row["mean_violation_depth"]) for row in items]),
        "MaxViolationDuration": np.mean([
            float(row["max_violation_duration_s"]) for row in items]),
    }


def baseline_rows() -> list[dict]:
    early = [row for row in load(
        ROOT / "development_adapter_tuning" / "development_runs.csv")
        if row["config"] == "baseline"]
    later = load(ROOT / "diagnostic_r1d_outputs" / "mujoco_runs.csv")
    return early + later


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "structural_runs.csv"
    existing = set()
    if raw.exists():
        existing = {(row["config"], int(row["seed"])) for row in load(raw)}
    predictor = core.NumpyDropoutBNN(seed=compatibility.TRAINING_SEED)
    predictor.fit_federated(seed=compatibility.TRAINING_SEED, actual_plant=True)
    for cap in CAPS:
        for soft in SOFT_WEIGHTS:
            for slack in COST_SLACKS:
                config = f"cap_{cap:.3f}_soft_{soft:.0f}_slack_{slack:.4f}"
                core.FED_TRUST_CAP = cap
                core.FED_SOFT_WEIGHT = soft
                core.FED_COST_SLACK = slack
                for seed in SEEDS:
                    if (config, seed) in existing:
                        continue
                    predictor.rng = np.random.default_rng(seed + 9000)
                    obstacles = core.generate_obstacles(seed)
                    xml = core.make_scene_xml(obstacles)
                    print(f"[structural development] {config} | seed={seed}", flush=True)
                    _, _, result = core.run_method(
                        "FedUMPC", xml, obstacles, predictor, show_viewer=False)
                    append(raw, {"config": config, "seed": seed, **asdict(result)})

    baseline = {}
    baseline_data = baseline_rows()
    for method in [value for value in core.METHODS if value != "FedUMPC"]:
        baseline[method] = aggregate([
            row for row in baseline_data if row["method"] == method])
    data = load(raw)
    summary = []
    for cap in CAPS:
        for soft in SOFT_WEIGHTS:
            for slack in COST_SLACKS:
                config = f"cap_{cap:.3f}_soft_{soft:.0f}_slack_{slack:.4f}"
                fed = aggregate([row for row in data if row["config"] == config])
                comparison = {**baseline, "FedUMPC": fed}
                best_count = 0
                rank_sum = 0.0
                for metric, direction in METRICS.items():
                    ordered = sorted(
                        comparison, key=lambda name: comparison[name][metric],
                        reverse=(direction == "max"))
                    values = [comparison[name][metric] for name in ordered]
                    fed_value = fed[metric]
                    best = values[0]
                    best_count += int(np.isclose(fed_value, best, atol=1e-12, rtol=0))
                    rank_sum += 1 + sum(
                        not np.isclose(value, fed_value, atol=1e-12, rtol=0)
                        for value in values if (value > fed_value if direction == "max"
                                                else value < fed_value))
                summary.append({
                    "config": config, "cap": cap, "soft_weight": soft,
                    "cost_slack": slack, "BestCount": best_count,
                    "MeanRank": rank_sum / len(METRICS), **fed,
                })
    summary.sort(key=lambda row: (-row["BestCount"], row["MeanRank"],
                                  -row["CBS"], -row["SR"], row["CR"]))
    with (OUT / "structural_summary.csv").open(
            "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    selection = {
        "development_seeds": SEEDS,
        "selection_rule": (
            "maximize seven-metric first-place count, minimize mean rank, "
            "then maximize CBS and SR and minimize CR"),
        "selected": summary[0],
        "baseline_metrics": baseline,
    }
    (OUT / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
