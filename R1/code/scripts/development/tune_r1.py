"""Development-seed-only calibration after the R1 obstacle-cost correction."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from r1_experiments import (Config, CrossingEnv, D_OBS, FedUMPCController,
                            MODEL_DIR, OUT, load_model, reset_rng,
                            run_simulation, temporary_config)
import pandas as pd


CONFIGS = [
    dict(label="A", uncertainty_weight=12.0, margin_scale=0.25,
         trust_scale=0.55, influence=4.2, soft_weight=0.02),
    dict(label="B", uncertainty_weight=26.0, margin_scale=0.25,
         trust_scale=0.55, influence=4.2, soft_weight=0.02),
    dict(label="C", uncertainty_weight=26.0, margin_scale=0.36,
         trust_scale=0.55, influence=4.2, soft_weight=0.02),
    dict(label="D", uncertainty_weight=40.0, margin_scale=0.36,
         trust_scale=0.55, influence=4.2, soft_weight=0.02),
    dict(label="E", uncertainty_weight=26.0, margin_scale=0.50,
         trust_scale=0.55, influence=4.2, soft_weight=0.02),
    dict(label="F", uncertainty_weight=26.0, margin_scale=0.36,
         trust_scale=0.40, influence=4.2, soft_weight=0.02),
    dict(label="G", uncertainty_weight=26.0, margin_scale=0.36,
         trust_scale=0.80, influence=4.2, soft_weight=0.02),
    dict(label="H", uncertainty_weight=26.0, margin_scale=0.36,
         trust_scale=0.55, influence=3.8, soft_weight=0.015),
]
SEEDS = [11, 23, 37]


def main():
    Config.set_mode("fast")
    prox = load_model(MODEL_DIR / "prox.pt")
    rows = []
    for config in CONFIGS:
        for seed in SEEDS:
            print(config["label"], seed, flush=True)
            reset_rng(seed)
            with temporary_config(
                    FED_SOFT_OBS_WEIGHT=config["soft_weight"],
                    FED_OBSTACLE_INFLUENCE=config["influence"],
                    OBSTACLE_PENALTY=7800.0,
                    UNCERTAINTY_WEIGHT=config["uncertainty_weight"],
                    UNCERTAINTY_MARGIN_SCALE=config["margin_scale"],
                    BNN_TRUST_UNC_SCALE=config["trust_scale"],
                    FED_TRACKING_BOOST=1.0,
                    FED_TERMINAL_BOOST=1.0):
                env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
                result = run_simulation(
                    [FedUMPCController(env, prox) for _ in range(4)],
                    env, seed=seed)
            rows.append({
                "Config": config["label"],
                "uncertainty_weight": config["uncertainty_weight"],
                "margin_scale": config["margin_scale"],
                "trust_scale": config["trust_scale"],
                "influence": config["influence"],
                "soft_weight": config["soft_weight"],
                "Seed": seed,
                "SR": result["SR"], "SC": result["SC"],
                "CR": result["CR"], "AvgDist": result["AvgDist"],
                "Cost": result["Cost"],
                "Score": result["SR"] + result["SC"] / 200.0
                         - result["CR"] / 2.0,
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "r1_development_calibration.csv", index=False)
    print(frame.groupby(["Config", "uncertainty_weight", "margin_scale",
                         "trust_scale", "influence", "soft_weight"])
               .mean(numeric_only=True)
               .sort_values("Score", ascending=False), flush=True)


if __name__ == "__main__":
    main()
