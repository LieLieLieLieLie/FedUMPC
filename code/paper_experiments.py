"""Reproducible experiments reported in the IoTJ manuscript.

The script reuses the paper's fast numerical configuration but evaluates the
main closed-loop benchmark on five fixed rollout seeds.  It additionally
tests (i) uncertainty/error consistency, (ii) matched local/FedAvg/Prox-FL
dynamics learners under the same uncertainty-aware controller, and (iii)
partial client participation.  Outputs are CSV/PDF/JSON so the provenance of
every value used in the revised manuscript remains auditable.
"""

from __future__ import annotations

import json
import inspect
import os
import platform
import random as py_random
import sys
import time
import types
from pathlib import Path

# Some Windows hosts expose a broken legacy crypto provider to Python's
# os.urandom.  These experiments are intentionally deterministic, so install a
# reproducible fallback before importing NumPy when the provider is unavailable.
try:
    os.urandom(1)
except OSError:
    _fallback_rng = py_random.Random(20260811)

    def _deterministic_urandom(n):
        return bytes(_fallback_rng.getrandbits(8) for _ in range(n))

    os.urandom = _deterministic_urandom
    py_random._urandom = _deterministic_urandom

# A separate host issue can prevent Windows' socket provider from initializing
# when Python imports asyncio through unittest.mock (which Torch imports).  The
# numerical code uses no asynchronous I/O, so a minimal coroutine-introspection
# shim is sufficient and keeps the experiment deterministic.
try:
    import _overlapped  # noqa: F401
except OSError:
    _asyncio_stub = types.ModuleType("asyncio")
    _asyncio_stub.iscoroutinefunction = inspect.iscoroutinefunction
    _asyncio_stub.coroutines = types.SimpleNamespace(_is_coroutine=object())
    sys.modules["asyncio"] = _asyncio_stub

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from config import Config
from controllers import (FedRMPCController, GP_MPC, LinearMPC, RobustMPC,
                         StochasticMPC, TubeMPC)
from federated import (DPFedAvgClient, DPProxFLClient, FedAvgClient,
                       FedServer, LocalClient, MOONClient, ProxFLClient)
from main import (D_OBS, MASS_DIST, _gen_data, _multi_seed, run_simulation)
from models import BNN
from utils import (CrossingEnv, VehicleModel, plot_ablation_efficiency,
                   plot_ablation_reliability, plot_fl_curves,
                   plot_privacy_tradeoff, plot_radar_chart_7,
                   plot_robustness_bars, plot_sensitivity)


OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)
Config.RESULTS_DIR = str(OUT)
ROLLOUT_SEEDS = [42, 123, 456, 789, 1024]
PARTICIPATION_SEEDS = ROLLOUT_SEEDS


def reset_rng(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def attach_data(clients, data) -> None:
    for i, client in enumerate(clients):
        for sample in data[i]:
            client.add_data(*sample)


def global_mse(model: BNN, data) -> float:
    states, actions, deltas = [], [], []
    for samples in data.values():
        for state, action, next_state in samples:
            states.append(state)
            actions.append(action)
            deltas.append(next_state - state)
    xu = torch.as_tensor(np.c_[states, actions], dtype=torch.float32,
                         device=Config.DEVICE)
    y = torch.as_tensor(np.asarray(deltas), dtype=torch.float32,
                        device=Config.DEVICE)
    model.eval()
    with torch.no_grad():
        return float(torch.mean((model(xu) - y) ** 2).item())


def train_global(client_cls, data, seed=666, participation=1.0):
    reset_rng(seed)
    server = FedServer()
    clients = [client_cls(i, mass=MASS_DIST[i]) for i in range(4)]
    attach_data(clients, data)
    weights = server.global_model.state_dict()
    rng = np.random.default_rng(seed + 1000)
    n_active = max(1, int(round(participation * len(clients))))
    aggregate_ms = []
    activations = np.zeros(len(clients), dtype=int)

    for _ in range(Config.ROUNDS):
        active_ids = (np.arange(len(clients)) if n_active == len(clients)
                      else np.sort(rng.choice(len(clients), n_active,
                                              replace=False)))
        active = [clients[int(i)] for i in active_ids]
        for i in active_ids:
            activations[int(i)] += 1
        for client in active:
            client.train(weights)
        start = time.perf_counter()
        weights = server.aggregate(active)
        aggregate_ms.append((time.perf_counter() - start) * 1e3)

    model = BNN().to(Config.DEVICE)
    model.load_state_dict(weights)
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    bidirectional_mb_per_client = (
        2.0 * model_bytes * float(np.mean(activations)) / 1e6
    )
    return model, {
        "final_mse": global_mse(model, data),
        "active_clients_per_round": n_active,
        "mean_aggregation_ms": float(np.mean(aggregate_ms)),
        "transmitted_mb_per_client": bidirectional_mb_per_client,
    }


def train_local_models(data, seed=666):
    reset_rng(seed)
    clients = [LocalClient(i, mass=MASS_DIST[i]) for i in range(4)]
    attach_data(clients, data)
    for _ in range(Config.ROUNDS):
        for client in clients:
            client.train_local()
    return [client.model for client in clients]


def aggregate_runs(ctrl_fn, seeds=ROLLOUT_SEEDS):
    env_fn = lambda s: CrossingEnv(seed=s, obstacle_density=D_OBS)
    return _multi_seed(ctrl_fn, env_fn, seeds=seeds)


def compact_metrics(name, agg):
    return {
        "Method": name,
        "SR_mean_pct": 100 * agg["SR_mean"],
        "SR_std_pct": 100 * agg["SR_std"],
        "Margin_mean_m": agg["AvgDist_mean"],
        "Margin_std_m": agg["AvgDist_std"],
        "SC_mean_pct": agg["SC_mean"],
        "SC_std_pct": agg["SC_std"],
        "Collision_mean": agg["CR_mean"],
        "Collision_std": agg["CR_std"],
        "Cost_mean": agg["Cost_mean"],
        "Cost_std": agg["Cost_std"],
        "Jerk_mean": agg["Jerk_mean"],
        "Jerk_std": agg["Jerk_std"],
        "Mean_controller_ms": agg["ComputeMs_mean"],
        "Mean_run_p95_ms": agg["ComputeP95Ms_mean"],
    }


def run_five_seed_main(prox_model):
    methods = {
        "Linear MPC": lambda env: [LinearMPC(env) for _ in range(4)],
        "Tube MPC": lambda env: [TubeMPC(env) for _ in range(4)],
        "Stochastic MPC": lambda env: [StochasticMPC(env) for _ in range(4)],
        "GP-MPC": lambda env: [GP_MPC(env) for _ in range(4)],
        "Robust MPC": lambda env: [RobustMPC(env) for _ in range(4)],
        "FedRMPC": lambda env: [FedRMPCController(env, prox_model)
                                 for _ in range(4)],
    }
    results = {}
    rows = []
    for name, fn in methods.items():
        print(f"[5-seed main] {name}", flush=True)
        results[name] = aggregate_runs(fn)
        rows.append(compact_metrics(name, results[name]))
    pd.DataFrame(rows).to_csv(OUT / "exp1_metrics_5seed.csv", index=False)
    pd.DataFrame([
        {"Method": row["Method"],
         "Mean_controller_ms": row["Mean_controller_ms"],
         "Mean_run_p95_ms": row["Mean_run_p95_ms"]}
        for row in rows
    ]).to_csv(OUT / "exp10_controller_latency.csv", index=False)
    plot_radar_chart_7(results, "exp1_mpc_bars.pdf")
    return results


def run_fair_fl_dynamics(data, seed=666):
    """Main-text learning comparison without an untuned SCAFFOLD row.

    The four retained learners share the same initialization, data, rounds,
    local epochs, batch size, and base learning rate.  Local-only drift is
    undefined because no global reference exists and is stored as NaN/N/A.
    """
    specs = [
        ("Local Only", LocalClient),
        ("FedAvg", FedAvgClient),
        ("MOON", MOONClient),
        ("Prox-FL", ProxFLClient),
    ]
    rounds = list(range(1, Config.ROUNDS + 1))
    history = {"Round": rounds}
    summary = []
    for name, client_cls in specs:
        print(f"[fair FL dynamics] {name}", flush=True)
        reset_rng(seed)
        server = FedServer()
        clients = [client_cls(i, mass=MASS_DIST[i]) for i in range(4)]
        attach_data(clients, data)
        weights = server.global_model.state_dict()
        mses, drifts = [], []
        for _ in rounds:
            local_mses = []
            for client in clients:
                if client_cls is LocalClient:
                    mse, _, _ = client.train_local()
                else:
                    mse, _, _ = client.train(weights)
                local_mses.append(float(mse))
            if client_cls is LocalClient:
                mses.append(float(np.mean(local_mses)))
                drifts.append(np.nan)
            else:
                weights = server.aggregate(clients)
                eval_model = BNN().to(Config.DEVICE)
                eval_model.load_state_dict(weights)
                mses.append(global_mse(eval_model, data))
                drifts.append(float(server.client_drift(clients)))
        history[f"{name}_MSE"] = mses
        history[f"{name}_Drift"] = drifts
        summary.append({
            "Method": name,
            "Final_MSE": mses[-1],
            "Final_Drift": np.nan if client_cls is LocalClient else drifts[-1],
        })

    history_df = pd.DataFrame(history)
    summary_df = pd.DataFrame(summary)
    history_df.to_csv(OUT / "exp2_fl_convergence_fair.csv", index=False)
    summary_df.to_csv(OUT / "exp2_fl_summary_fair.csv", index=False)
    plot_fl_curves(history, [name for name, _ in specs],
                   "exp2_training_dynamics.pdf")
    return summary_df


def run_five_seed_robustness(prox_model):
    scenarios = [
        ("Low Noise", {"noise": 0.05}),
        ("High Noise", {"noise": 0.25}),
        ("Wind", {"disturbance": 1.0}),
        ("Model Error", {"mass_error": 0.5}),
    ]
    methods = {
        "Linear MPC": lambda env: [LinearMPC(env) for _ in range(4)],
        "Tube MPC": lambda env: [TubeMPC(env) for _ in range(4)],
        "Stochastic MPC": lambda env: [StochasticMPC(env) for _ in range(4)],
        "GP-MPC": lambda env: [GP_MPC(env) for _ in range(4)],
        "Robust MPC": lambda env: [RobustMPC(env) for _ in range(4)],
        "FedRMPC": lambda env: [FedRMPCController(env, prox_model)
                                  for _ in range(4)],
    }
    env_fn = lambda s: CrossingEnv(seed=s, obstacle_density=D_OBS)
    rows = []
    for scenario, kwargs in scenarios:
        for method, controller_fn in methods.items():
            print(f"[5-seed robustness] {scenario} | {method}", flush=True)
            agg = _multi_seed(controller_fn, env_fn, seeds=ROLLOUT_SEEDS,
                              **kwargs)
            rows.append({
                "Scenario": scenario,
                "Method": method,
                "Seeds": ",".join(map(str, ROLLOUT_SEEDS)),
                "SR (%)": 100 * agg["SR_mean"],
                "SR std (%)": 100 * agg["SR_std"],
                "SC (%)": agg["SC_mean"],
                "SC std (%)": agg["SC_std"],
                "AvgDist": agg["AvgDist_mean"],
                "AvgDist std": agg["AvgDist_std"],
                "Collision": agg["CR_mean"],
                "Collision std": agg["CR_std"],
                "Cost": agg["Cost_mean"],
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp3_robustness_5seed.csv", index=False)
    plot_robustness_bars(frame, "exp3_robustness_metrics.pdf")
    return frame


def run_heterogeneous_plant(prox_model, metadata):
    """Five-seed closed-loop test with four genuinely different plants."""
    masses = [800.0, 900.0, 1000.0, 1100.0]
    drags = [float(metadata[i]["drag"]) for i in range(4)]
    methods = {
        "Tube MPC": lambda env: [TubeMPC(env) for _ in range(4)],
        "GP-MPC": lambda env: [GP_MPC(env) for _ in range(4)],
        "Robust MPC": lambda env: [RobustMPC(env) for _ in range(4)],
        "FedRMPC": lambda env: [FedRMPCController(env, prox_model)
                                  for _ in range(4)],
    }
    env_fn = lambda s: CrossingEnv(seed=s, obstacle_density=D_OBS)
    rows = []
    for method, controller_fn in methods.items():
        print(f"[heterogeneous plant] {method}", flush=True)
        agg = _multi_seed(controller_fn, env_fn, seeds=ROLLOUT_SEEDS,
                          vehicle_masses=masses, vehicle_drags=drags)
        row = compact_metrics(method, agg)
        row.update({
            "Seeds": ",".join(map(str, ROLLOUT_SEEDS)),
            "Vehicle_masses_kg": ",".join(f"{x:.0f}" for x in masses),
            "Vehicle_drags": ",".join(f"{x:.5f}" for x in drags),
        })
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp11_heterogeneous_plant.csv", index=False)

    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.8))
    names = list(frame["Method"])
    panels = [
        ("SR_mean_pct", "SR (%)"),
        ("SC_mean_pct", "SC (%)"),
        ("Margin_mean_m", "Margin (m)"),
        ("Collision_mean", "Mean contacts"),
    ]
    colors = ["#4C78A8", "#59A14F", "#F28E2B", "#B279A2"]
    x = np.arange(len(names))
    for ax, (key, ylabel) in zip(axes, panels):
        ax.bar(x, frame[key], color=colors, width=0.72)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=28, ha="right", fontsize=7)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(OUT / "exp11_heterogeneous_plant.pdf", bbox_inches="tight")
    plt.close(fig)
    return frame


def run_matched_learner_comparison(local_models, fedavg_model, prox_model,
                                   prox_reference=None):
    configs = {
        "Local only": lambda env: [FedRMPCController(env, local_models[i])
                                     for i in range(4)],
        "FedAvg": lambda env: [FedRMPCController(env, fedavg_model)
                                for _ in range(4)],
        "Prox-FL": lambda env: [FedRMPCController(env, prox_model)
                                 for _ in range(4)],
    }
    rows = []
    for name, fn in configs.items():
        if name == "Prox-FL" and prox_reference is not None:
            rows.append(compact_metrics(name, prox_reference))
            continue
        print(f"[matched learner] {name}", flush=True)
        # Pair the MC-dropout masks as well as the rollout/environment seeds.
        reset_rng(20260811)
        agg = aggregate_runs(fn)
        rows.append(compact_metrics(name, agg))
    pd.DataFrame(rows).to_csv(OUT / "exp8_matched_fl_controller.csv",
                              index=False)
    return rows


def uncertainty_error_samples(model, metadata, n_per_client=250):
    reset_rng(20260811)
    scenario_defs = {
        "Nominal": (0.0, 0.0, [1.0, 1.0, 1.0, 1.0]),
        "High noise": (0.25, 0.0, [1.0, 1.0, 1.0, 1.0]),
        "Wind": (0.0, 1.0, [1.0, 1.0, 1.0, 1.0]),
        "Model error": (0.0, 0.0, [1.5, 0.5, 1.0, 1.0]),
    }
    rng = np.random.default_rng(20260811)
    rows = []
    for scenario, (noise, disturbance, mass_scale) in scenario_defs.items():
        states, actions, targets, client_ids = [], [], [], []
        for i in range(4):
            plant = VehicleModel(mass=metadata[i]["mass_kg"] * mass_scale[i],
                                 drag=metadata[i]["drag"])
            for _ in range(n_per_client):
                state = np.array([rng.uniform(-22, 22), rng.uniform(-22, 22),
                                  rng.uniform(0, 12), rng.uniform(-np.pi, np.pi)])
                action = rng.uniform([-Config.MAX_STEER, -Config.MAX_ACCEL],
                                     [Config.MAX_STEER, Config.MAX_ACCEL])
                # VehicleModel uses NumPy's global RNG for process noise.
                np.random.seed(int(rng.integers(0, 2**31 - 1)))
                next_state = plant.step(state, action, noise_std=noise,
                                        disturbance=disturbance)
                states.append(state)
                actions.append(action)
                targets.append(next_state - state)
                client_ids.append(i + 1)

        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        means, variances = [], []
        for start in range(0, len(states), 500):
            mean, var = model.predict_horizon(states[start:start + 500],
                                              actions[start:start + 500])
            means.append(mean)
            variances.append(var)
        means = np.vstack(means)
        uncertainty = np.sqrt(np.maximum(np.concatenate(variances), 0.0))
        error = np.linalg.norm(means - targets, axis=1)
        for client, unc, err in zip(client_ids, uncertainty, error):
            rows.append({"Scenario": scenario, "Client": client,
                         "Uncertainty": float(unc),
                         "Prediction_error": float(err)})
    return pd.DataFrame(rows)


def plot_uncertainty_error(df):
    rho, pvalue = spearmanr(df["Uncertainty"], df["Prediction_error"])
    bins = pd.qcut(df["Uncertainty"], 4,
                   labels=["Q1 (lowest)", "Q2", "Q3", "Q4 (highest)"],
                   duplicates="drop")
    quartile = (df.assign(Uncertainty_quartile=bins)
                  .groupby("Uncertainty_quartile", observed=True)
                  .agg(Mean_error=("Prediction_error", "mean"),
                       Std_error=("Prediction_error", "std"),
                       N=("Prediction_error", "size"))
                  .reset_index())
    summary_rows = []
    for scenario, sub in df.groupby("Scenario", sort=False):
        r, p = spearmanr(sub["Uncertainty"], sub["Prediction_error"])
        summary_rows.append({"Scenario": scenario, "N": len(sub),
                             "Spearman_rho": r, "p_value": p,
                             "Mean_uncertainty": sub["Uncertainty"].mean(),
                             "Mean_prediction_error": sub["Prediction_error"].mean()})
    summary_rows.append({"Scenario": "Pooled", "N": len(df),
                         "Spearman_rho": rho, "p_value": pvalue,
                         "Mean_uncertainty": df["Uncertainty"].mean(),
                         "Mean_prediction_error": df["Prediction_error"].mean()})
    pd.DataFrame(summary_rows).to_csv(
        OUT / "exp7_uncertainty_error_summary.csv", index=False)
    quartile.to_csv(OUT / "exp7_uncertainty_error_quartiles.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
    hb = axes[0].hexbin(df["Uncertainty"], df["Prediction_error"], gridsize=40,
                        mincnt=1, cmap="Blues", linewidths=0)
    axes[0].set_xlabel(r"Epistemic uncertainty $\mathcal{U}$", fontsize=9)
    axes[0].set_ylabel(r"Actual one-step error $\|\hat{\Delta s}-\Delta s\|_2$",
                       fontsize=9)
    axes[0].set_title(f"(a) Pooled consistency ($\\rho_s={rho:.2f}$)",
                      fontsize=10)
    colorbar = fig.colorbar(hb, ax=axes[0])
    colorbar.set_label("Samples per bin", fontsize=8)
    colorbar.ax.tick_params(labelsize=8)

    x = np.arange(len(quartile))
    axes[1].bar(x, quartile["Mean_error"], color="#6699CC",
                yerr=quartile["Std_error"], capsize=4)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    axes[1].set_xlabel("Uncertainty quartile", fontsize=9)
    axes[1].set_ylabel("Mean actual one-step error", fontsize=9)
    axes[1].set_title("(b) Error by uncertainty quartile", fontsize=10)
    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(OUT / "exp7_uncertainty_error.pdf", bbox_inches="tight")
    plt.close(fig)


def run_participation_stress(data):
    rows = []
    for participation in [1.0, 0.75, 0.5]:
        print(f"[participation] p={participation}", flush=True)
        model, train_info = train_global(ProxFLClient, data, seed=777,
                                         participation=participation)
        agg = aggregate_runs(lambda env, m=model:
                             [FedRMPCController(env, m) for _ in range(4)],
                             seeds=PARTICIPATION_SEEDS)
        rows.append({
            "Participation_rate": participation,
            "Active_clients_per_round": train_info["active_clients_per_round"],
            "Communication_rounds": Config.ROUNDS,
            "Final_MSE": train_info["final_mse"],
            "SR_mean_pct": 100 * agg["SR_mean"],
            "SR_std_pct": 100 * agg["SR_std"],
            "SC_mean_pct": agg["SC_mean"],
            "SC_std_pct": agg["SC_std"],
            "Collision_mean": agg["CR_mean"],
            "Collision_std": agg["CR_std"],
            "Transmitted_MB_per_client": train_info["transmitted_mb_per_client"],
            "Mean_aggregation_ms": train_info["mean_aggregation_ms"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp9_participation_stress.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.0))
    labels = ["1.00", "0.75", "0.50"]
    panels = [("Final_MSE", "Validation MSE"),
              ("SR_mean_pct", "Success rate (%)"),
              ("SC_mean_pct", "Safety compliance (%)")]
    for ax, (key, ylabel) in zip(axes, panels):
        ax.plot(labels, frame[key], marker="o", linewidth=2.4,
                color="#7D4AB3")
        ax.set_xlabel("Participation rate", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.45)
    fig.tight_layout()
    fig.savefig(OUT / "exp9_participation_stress.pdf", bbox_inches="tight")
    plt.close(fig)
    return rows


def run_five_seed_ablation(local_models, prox_model, prox_reference=None):
    configs = {
        "Full FedRMPC": lambda env: [FedRMPCController(env, prox_model, True)
                                       for _ in range(4)],
        "w/o Uncertainty": lambda env: [FedRMPCController(env, prox_model, False)
                                          for _ in range(4)],
        "w/o Federated": lambda env: [FedRMPCController(env, local_models[i], True)
                                        for i in range(4)],
    }
    rows = []
    for name, controller_fn in configs.items():
        print(f"[5-seed ablation] {name}", flush=True)
        if name == "Full FedRMPC" and prox_reference is not None:
            agg = prox_reference
        else:
            reset_rng(20260811)
            agg = aggregate_runs(controller_fn)
        rows.append({
            "Config": name,
            "SR (%)": 100 * agg["SR_mean"],
            "SR std (%)": 100 * agg["SR_std"],
            "SC (%)": agg["SC_mean"],
            "SC std (%)": agg["SC_std"],
            "AvgDist": agg["AvgDist_mean"],
            "AvgDist std": agg["AvgDist_std"],
            "Cost": agg["Cost_mean"],
            "Cost std": agg["Cost_std"],
            "Jerk": agg["Jerk_mean"],
            "Jerk std": agg["Jerk_std"],
            "Collision": agg["CR_mean"],
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp4_ablation_5seed.csv", index=False)
    plot_ablation_reliability(frame, "exp4_ablation_reliability.pdf")
    plot_ablation_efficiency(frame, "exp4_ablation_efficiency.pdf")
    return frame


def _train_prox_with_mu(data, mu, seed):
    original_mu = Config.PROXIMAL_MU
    Config.PROXIMAL_MU = float(mu)
    try:
        model, _ = train_global(ProxFLClient, data, seed=seed)
    finally:
        Config.PROXIMAL_MU = original_mu
    return model


def run_five_seed_sensitivity(data, prox_model):
    mu_values = [0.04, 0.10, 0.20]
    mu_rows = []
    for mu in mu_values:
        print(f"[5-seed sensitivity] mu={mu}", flush=True)
        model = (prox_model if np.isclose(mu, 0.10)
                 else _train_prox_with_mu(data, mu, seed=666))
        reset_rng(20260811)
        agg = aggregate_runs(lambda env, m=model:
                             [FedRMPCController(env, m) for _ in range(4)])
        mu_rows.append({"mu": mu, "SR": agg["SR_mean"],
                        "SR_std": agg["SR_std"], "SC": agg["SC_mean"],
                        "SC_std": agg["SC_std"]})

    lambda_values = [10.0, 18.0, 28.0]
    lambda_rows = []
    for value in lambda_values:
        print(f"[5-seed sensitivity] lambda={value}", flush=True)
        reset_rng(20260811)
        agg = aggregate_runs(
            lambda env, v=value: [FedRMPCController(env, prox_model, lambda0=v)
                                  for _ in range(4)])
        lambda_rows.append({"lambda": value, "SR": agg["SR_mean"],
                            "SR_std": agg["SR_std"], "SC": agg["SC_mean"],
                            "SC_std": agg["SC_std"]})

    mu_frame = pd.DataFrame(mu_rows)
    lambda_frame = pd.DataFrame(lambda_rows)
    mu_frame.to_csv(OUT / "exp5_sens_mu_5seed.csv", index=False)
    lambda_frame.to_csv(OUT / "exp5_sens_lambda_5seed.csv", index=False)
    plot_sensitivity({
        "mu_values": mu_values,
        "mu_sr": list(mu_frame["SR"]),
        "mu_sc": list(mu_frame["SC"]),
        "lambda_values": lambda_values,
        "lam_sr": list(lambda_frame["SR"]),
        "lam_sc": list(lambda_frame["SC"]),
    }, "exp5_sensitivity.pdf")
    return mu_frame, lambda_frame


def train_noisy_global(client_cls, data, noise_mult, seed=666):
    reset_rng(seed)
    server = FedServer()
    clients = [client_cls(i, mass=MASS_DIST[i], noise_mult=noise_mult)
               for i in range(4)]
    attach_data(clients, data)
    weights = server.global_model.state_dict()
    for _ in range(max(6, Config.ROUNDS // 2)):
        for client in clients:
            client.train(weights)
        weights = server.aggregate(clients)
    model = BNN().to(Config.DEVICE)
    model.load_state_dict(weights)
    return model, global_mse(model, data)


def train_short_clean_global(client_cls, data, seed=666):
    reset_rng(seed)
    server = FedServer()
    clients = [client_cls(i, mass=MASS_DIST[i]) for i in range(4)]
    attach_data(clients, data)
    weights = server.global_model.state_dict()
    for _ in range(max(6, Config.ROUNDS // 2)):
        for client in clients:
            client.train(weights)
        weights = server.aggregate(clients)
    model = BNN().to(Config.DEVICE)
    model.load_state_dict(weights)
    return model, global_mse(model, data)


def run_five_seed_gradient_noise(data):
    settings = [("High", 2.018), ("Moderate", 0.571), ("None", 0.0)]
    rows = []
    for setting, noise in settings:
        for method, noisy_cls, clean_cls in [
            ("DP-FedAvg", DPFedAvgClient, FedAvgClient),
            ("DP-Prox-FL", DPProxFLClient, ProxFLClient),
        ]:
            print(f"[5-seed gradient noise] {setting} | {method}", flush=True)
            if np.isclose(noise, 0.0):
                model, final_mse = train_short_clean_global(
                    clean_cls, data, seed=666)
                display_method = method.replace("DP-", "")
            else:
                model, final_mse = train_noisy_global(noisy_cls, data, noise)
                display_method = method
            reset_rng(20260811)
            agg = aggregate_runs(lambda env, m=model:
                                 [FedRMPCController(env, m) for _ in range(4)])
            rows.append({
                "Method": display_method,
                "Noise setting": setting,
                "Noise mult": noise,
                "Final MSE": final_mse,
                "SR (%)": 100 * agg["SR_mean"],
                "SR std (%)": 100 * agg["SR_std"],
                "SC (%)": agg["SC_mean"],
                "SC std (%)": agg["SC_std"],
                "Collision": agg["CR_mean"],
            })
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "exp6_gradient_noise_5seed.csv", index=False)
    # The plotting helper expects the legacy method labels for all settings.
    plot_frame = frame.copy()
    plot_frame.loc[(plot_frame["Noise setting"] == "None") &
                   (plot_frame["Method"] == "FedAvg"), "Method"] = "DP-FedAvg"
    plot_frame.loc[(plot_frame["Noise setting"] == "None") &
                   (plot_frame["Method"] == "Prox-FL"), "Method"] = "DP-Prox-FL"
    plot_privacy_tradeoff(plot_frame, "exp6_privacy_tradeoff.pdf")
    return frame


def hardware_metadata():
    cpu_name = platform.processor() or platform.machine()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 2**30
    except Exception:
        ram_gb = None
    return {
        "cpu": cpu_name,
        "ram_gb": ram_gb,
        "device": str(Config.DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "solver": "finite multi-start candidate evaluation; SLSQP disabled",
    }


def main():
    Config.set_mode("fast")
    Config.SEEDS = ROLLOUT_SEEDS
    Config.N_SEEDS = len(ROLLOUT_SEEDS)
    reset_rng(100)
    data, metadata = _gen_data(return_metadata=True)
    (OUT / "exp_metadata.json").write_text(json.dumps({
        "rollout_seeds": ROLLOUT_SEEDS,
        "participation_stress_seed": PARTICIPATION_SEEDS,
        "training_clients": metadata,
        "hardware": hardware_metadata(),
        "configuration": {
            "rounds": Config.ROUNDS, "local_epochs": Config.LOCAL_EPOCHS,
            "batch_size": Config.BATCH_SIZE, "samples_per_client": Config.N_DATA,
            "horizon": Config.HORIZON, "mc_samples": Config.MC_SAMPLES,
            "rollout_rng_protocol": (
                "NumPy, PyTorch CPU, and all CUDA generators are reset to "
                "the listed rollout seed at the start of every run"
            ),
        },
    }, indent=2), encoding="utf-8")

    print("[training] FedAvg", flush=True)
    fedavg_model, _ = train_global(FedAvgClient, data, seed=666)
    print("[training] Prox-FL", flush=True)
    prox_model, _ = train_global(ProxFLClient, data, seed=666)
    print("[training] Local-only models", flush=True)
    local_models = train_local_models(data, seed=666)

    run_fair_fl_dynamics(data)
    main_results = run_five_seed_main(prox_model)
    run_five_seed_robustness(prox_model)
    run_heterogeneous_plant(prox_model, metadata)
    run_matched_learner_comparison(local_models, fedavg_model, prox_model,
                                   prox_reference=main_results["FedRMPC"])

    uncertainty_df = uncertainty_error_samples(prox_model, metadata)
    uncertainty_df.to_csv(OUT / "exp7_uncertainty_error_samples.csv", index=False)
    plot_uncertainty_error(uncertainty_df)
    run_participation_stress(data)
    print(f"[done] manuscript experiments -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
