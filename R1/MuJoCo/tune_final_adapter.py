"""Select the final MuJoCo adapter on development scenes only."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

import r1_mujoco_final as compatibility
import run_six_algorithms as core


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "development_adapter_tuning"
SEEDS = [11, 23, 37]
CAPS = [0.05, 0.10, 0.20, 0.35, 0.55]
SOFT_WEIGHTS = [75.0, 150.0, 250.0]


def append(path: Path, row: dict) -> None:
    header = not path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if header:
            writer.writeheader()
        writer.writerow(row)


def rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def aggregate(items: list[dict]) -> dict[str, float]:
    success = np.mean([float(row["success"]) for row in items]) / 4 * 100
    collision = np.mean([float(row["collisions"]) for row in items]) / 4 * 100
    vf = np.mean([float(row["violation_frequency_pct"]) for row in items])
    sc = 100 - vf
    cbs = 2 * success * sc / max(success + sc, 1e-12) * (1 - collision / 100)
    return {
        "SR": success, "SC": sc, "CR": collision, "CBS": cbs,
        "MinClearance": np.mean([float(row["min_clearance"]) for row in items]),
        "MeanViolationDepth": np.mean([
            float(row["mean_violation_depth"]) for row in items]),
        "MaxViolationDuration": np.mean([
            float(row["max_violation_duration_s"]) for row in items]),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "development_runs.csv"
    predictor = core.NumpyDropoutBNN(seed=compatibility.TRAINING_SEED)
    predictor.fit_federated(seed=compatibility.TRAINING_SEED, actual_plant=True)
    completed = {
        (row["config"], int(row["seed"]), row["method"]) for row in rows(raw)
    }
    baseline_methods = [method for method in core.METHODS if method != "FedUMPC"]
    for seed in SEEDS:
        obstacles = core.generate_obstacles(seed)
        xml = core.make_scene_xml(obstacles)
        for method in baseline_methods:
            key = ("baseline", seed, method)
            if key in completed:
                continue
            print(f"[adapter development] baseline {method} | seed={seed}", flush=True)
            _, _, result = core.run_method(
                method, xml, obstacles, predictor, show_viewer=False)
            append(raw, {"config": "baseline", "seed": seed, **asdict(result)})
    for cap in CAPS:
        for soft in SOFT_WEIGHTS:
            config = f"cap_{cap:.2f}_soft_{soft:.0f}"
            core.FED_TRUST_CAP = cap
            core.FED_SOFT_WEIGHT = soft
            for seed in SEEDS:
                key = (config, seed, "FedUMPC")
                if key in completed:
                    continue
                predictor.rng = np.random.default_rng(seed + 9000)
                obstacles = core.generate_obstacles(seed)
                xml = core.make_scene_xml(obstacles)
                print(f"[adapter development] {config} | seed={seed}", flush=True)
                _, _, result = core.run_method(
                    "FedUMPC", xml, obstacles, predictor, show_viewer=False)
                append(raw, {"config": config, "seed": seed, **asdict(result)})

    data = rows(raw)
    baselines = {
        method: aggregate([row for row in data
                           if row["config"] == "baseline" and row["method"] == method])
        for method in baseline_methods
    }
    metrics = {
        "SR": "max", "SC": "max", "CR": "min", "CBS": "max",
        "MinClearance": "max", "MeanViolationDepth": "min",
        "MaxViolationDuration": "min",
    }
    summary = []
    for cap in CAPS:
        for soft in SOFT_WEIGHTS:
            config = f"cap_{cap:.2f}_soft_{soft:.0f}"
            fed = aggregate([row for row in data if row["config"] == config])
            best_count = 0
            for metric, direction in metrics.items():
                values = [entry[metric] for entry in baselines.values()] + [fed[metric]]
                target = max(values) if direction == "max" else min(values)
                best_count += int(np.isclose(fed[metric], target, atol=1e-12, rtol=0))
            summary.append({"config": config, "cap": cap, "soft_weight": soft,
                            "BestCount": best_count, **fed})
    summary.sort(key=lambda row: (-row["BestCount"], -row["CBS"],
                                  -row["SR"], row["CR"]))
    with (OUT / "development_summary.csv").open(
            "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    selection = {
        "development_seeds": SEEDS,
        "selection_rule": "maximize seven-metric first-place count, then CBS, SR, and lower CR",
        "selected": summary[0],
        "baseline_metrics": baselines,
    }
    (OUT / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
