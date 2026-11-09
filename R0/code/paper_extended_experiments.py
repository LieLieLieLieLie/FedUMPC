"""Run the extended five-seed ablation and diagnostic experiments."""

from paper_experiments import (
    OUT,
    _gen_data,
    reset_rng,
    train_global,
    train_local_models,
    run_five_seed_ablation,
    run_five_seed_gradient_noise,
    run_five_seed_sensitivity,
    uncertainty_error_samples,
    plot_uncertainty_error,
    pd,
)
from config import Config
from federated import ProxFLClient


def main():
    Config.set_mode("fast")
    reset_rng(100)
    data, metadata = _gen_data(return_metadata=True)
    print("[training] shared Prox-FL model", flush=True)
    prox_model, _ = train_global(ProxFLClient, data, seed=666)
    print("[training] vehicle-local models", flush=True)
    local_models = train_local_models(data, seed=666)

    run_five_seed_ablation(local_models, prox_model)
    run_five_seed_sensitivity(data, prox_model)
    run_five_seed_gradient_noise(data)
    uncertainty = uncertainty_error_samples(prox_model, metadata)
    uncertainty.to_csv(OUT / "exp7_uncertainty_error_samples.csv", index=False)
    plot_uncertainty_error(uncertainty)
    print(f"[done] final five-seed experiments -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
