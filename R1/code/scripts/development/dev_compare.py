"""Development-seed comparison for the frozen R1 configuration."""

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

from r1_experiments import (Config, CrossingEnv, D_OBS, FedUMPCController,
                            MODEL_DIR, RobustMPC, TubeMPC, load_model,
                            run_simulation)
import pandas as pd


def main():
    Config.set_mode("fast")
    prox = load_model(MODEL_DIR / "prox.pt")
    central = load_model(MODEL_DIR / "central.pt")
    methods = {
        "Tube-style MPC": lambda env: [TubeMPC(env) for _ in range(4)],
        "Fixed-margin MPC": lambda env: [RobustMPC(env) for _ in range(4)],
        "Centralized UMPC": lambda env: [FedUMPCController(env, central)
                                          for _ in range(4)],
        "FedUMPC": lambda env: [FedUMPCController(env, prox)
                                 for _ in range(4)],
    }
    rows = []
    for name, factory in methods.items():
        for seed in [11, 23, 37]:
            print(name, seed, flush=True)
            env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
            result = run_simulation(factory(env), env, seed=seed)
            rows.append({"Method": name, "Seed": seed,
                         "SR": result["SR"], "SC": result["SC"],
                         "CR": result["CR"],
                         "MinClearance": result["MinClearance"]})
    frame = pd.DataFrame(rows)
    frame.to_csv("results_r1/r1_development_method_comparison.csv", index=False)
    print(frame.groupby("Method").mean(numeric_only=True), flush=True)


if __name__ == "__main__":
    main()
