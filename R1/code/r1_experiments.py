"""Reviewer-driven R1 experiments for the IoTJ FedUMPC resubmission.

The script is intentionally checkpointed: every completed seed is appended to
CSV, and trained models are cached under ``results_r1/models``.  Re-running the
script resumes missing work without replacing completed measurements.
"""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import inspect
import json
import math
import os
import platform
import random as py_random
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

# Deterministic fallbacks for the two broken Windows providers observed on the
# experiment workstation.  They are installed before NumPy/Torch import.
try:
    os.urandom(1)
except OSError:
    _fallback_rng = py_random.Random(20261103)

    def _deterministic_urandom(n):
        return bytes(_fallback_rng.getrandbits(8) for _ in range(n))

    os.urandom = _deterministic_urandom
    py_random._urandom = _deterministic_urandom

try:
    import _overlapped  # noqa: F401
except OSError:
    _asyncio_stub = types.ModuleType("asyncio")
    _asyncio_stub.iscoroutinefunction = inspect.iscoroutinefunction
    _asyncio_stub.coroutines = types.SimpleNamespace(_is_coroutine=object())
    sys.modules["asyncio"] = _asyncio_stub

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import binomtest, spearmanr, t

from config import Config
from controllers import (FedUMPCController, GP_MPC, LinearMPC, RobustMPC,
                         StochasticMPC, TubeMPC)
from federated import FedAvgClient, FedServer, LocalClient, ProxFLClient
from main import D_OBS, MASS_DIST, _gen_data, run_simulation
from models import BNN, EnsemblePredictor
from utils import CrossingEnv, VehicleModel


ROOT = R1_ROOT
OUT = artifact_path("results_r1")
MODEL_DIR = OUT / "models"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
Config.RESULTS_DIR = str(OUT)

TEST_SEEDS = [42, 123, 456, 789, 1024, 2048, 4096, 8192, 16384, 32768]
MC_TEST_SEEDS = TEST_SEEDS[:5]
SENSITIVITY_SEEDS = [11, 23, 37]
FEDERATION_SEEDS = [2026, 2027, 2028]

FED_RED = "#FF6666"
PALETTE = ["#FFAA53", "#50CC55", "#3399FF", "#6666FF",
           "#9933FF", "#00DDDD", "#4D4D4D"]
DIVERGING = mcolors.LinearSegmentedColormap.from_list(
    "review_diverging", ["#007FFF", "#FFFFFF", "#FF4F4F"])
POSITIVE = mcolors.LinearSegmentedColormap.from_list(
    "review_positive", ["#FFFFFF", "#FF4F4F"])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8.2,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def reset_rng(seed: int) -> None:
    np.random.seed(seed)
    py_random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def temporary_config(**updates):
    old = {name: getattr(Config, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(Config, name, value)
        yield
    finally:
        for name, value in old.items():
            setattr(Config, name, value)


def append_row(path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    if path.exists():
        # Preserve the checkpoint schema when optional diagnostics are added.
        # Otherwise positional appends silently shift later columns.
        columns = pd.read_csv(path, nrows=0).columns
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing checkpoint columns: {sorted(missing)}")
        frame = frame.reindex(columns=columns)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)


def completed_keys(path: Path, columns: list[str]) -> set[tuple]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    return {tuple(row[c] for c in columns) for _, row in frame.iterrows()}


def model_bytes(model) -> int:
    if isinstance(model, EnsemblePredictor):
        return sum(model_bytes(member) for member in model.members)
    return sum(p.numel() * p.element_size() for p in model.parameters())


def attach_data(clients, data) -> None:
    for i, client in enumerate(clients):
        for sample in data[i]:
            client.add_data(*sample)


def mse_model(model, samples) -> float:
    states = np.asarray([s for s, _, _ in samples], dtype=np.float32)
    actions = np.asarray([u for _, u, _ in samples], dtype=np.float32)
    targets = np.asarray([sn - s for s, _, sn in samples], dtype=np.float32)
    xu = torch.as_tensor(np.c_[states, actions], dtype=torch.float32,
                         device=Config.DEVICE)
    model.eval()
    with torch.no_grad():
        pred = model(xu).cpu().numpy()
    scale = model.delta_scale.detach().cpu().numpy()
    return float(np.mean(((pred - targets) / scale) ** 2))


def train_federated(client_cls, data, seed=666, rounds=None,
                    participation=1.0, staleness=0):
    reset_rng(seed)
    rounds = Config.ROUNDS if rounds is None else int(rounds)
    server = FedServer()
    # Match Adam and its learning rate across FedAvg and Prox-FL.  The
    # legacy FedAvg class used SGD, which confounded the proximal comparison.
    class MatchedFedAvgClient(ProxFLClient):
        def train(self, global_weights, **kw):
            with temporary_config(PROXIMAL_MU=0.0):
                return super().train(global_weights, **kw)
    effective_cls = (MatchedFedAvgClient if client_cls is FedAvgClient
                     else client_cls)
    clients = [effective_cls(i, mass=800.0 + 20.0 * i) for i in range(len(data))]
    attach_data(clients, data)
    weights = server.global_model.state_dict()
    history = [{k: v.detach().clone() for k, v in weights.items()}]
    rng = np.random.default_rng(seed + 991)
    final_drift = np.nan
    for round_index in range(rounds):
        n_active = max(1, int(round(participation * len(clients))))
        ids = (np.arange(len(clients)) if n_active == len(clients)
               else np.sort(rng.choice(len(clients), n_active, replace=False)))
        stale_index = max(0, len(history) - 1 - int(staleness))
        broadcast = history[stale_index]
        active = [clients[int(i)] for i in ids]
        for client in active:
            client.train(broadcast)
        final_drift = float(np.mean([
            sum((value.float() - broadcast[key].float()).norm(2).item()
                for key, value in client.model.state_dict().items())
            for client in active]))
        weights = server.aggregate(active)
        history.append({k: v.detach().clone() for k, v in weights.items()})
    model = BNN().to(Config.DEVICE)
    model.load_state_dict(weights)
    return model, clients, final_drift


def train_local(data, seed=666, rounds=None):
    reset_rng(seed)
    rounds = Config.ROUNDS if rounds is None else int(rounds)
    clients = [LocalClient(i, mass=800.0 + 20.0 * i)
               for i in range(len(data))]
    attach_data(clients, data)
    for _ in range(rounds):
        for client in clients:
            with temporary_config(LR=Config.LR * Config.PROX_FL_LR_MULT):
                client.train_local()
    return [client.model for client in clients]


def train_centralized(data, seed=666, epochs=90):
    reset_rng(seed)
    model = BNN().to(Config.DEVICE)
    samples = [sample for client in data.values() for sample in client]
    states = np.asarray([s for s, _, _ in samples], dtype=np.float32)
    actions = np.asarray([u for _, u, _ in samples], dtype=np.float32)
    targets = np.asarray([sn - s for s, _, sn in samples], dtype=np.float32)
    xu = torch.as_tensor(np.c_[states, actions], dtype=torch.float32,
                         device=Config.DEVICE)
    y = torch.as_tensor(targets, dtype=torch.float32, device=Config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=Config.LR * Config.PROX_FL_LR_MULT)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(int(epochs)):
        indices = torch.randperm(len(xu), generator=generator)
        model.train()
        for start in range(0, len(xu), Config.BATCH_SIZE):
            batch = indices[start:start + Config.BATCH_SIZE].to(Config.DEVICE)
            optimizer.zero_grad()
            loss = torch.mean((model(xu[batch]) - y[batch]) ** 2)
            loss.backward()
            optimizer.step()
    return model


def save_model(path: Path, model: BNN) -> None:
    torch.save(model.state_dict(), path)


def load_model(path: Path) -> BNN:
    model = BNN().to(Config.DEVICE)
    model.load_state_dict(torch.load(path, map_location=Config.DEVICE))
    return model


def prepare_primary_models():
    metadata_path = MODEL_DIR / "training_metadata.json"
    prox_path = MODEL_DIR / "prox.pt"
    fedavg_path = MODEL_DIR / "fedavg.pt"
    central_path = MODEL_DIR / "central.pt"
    local_paths = [MODEL_DIR / f"local_{i}.pt" for i in range(4)]
    ensemble_paths = [MODEL_DIR / f"ensemble_{i}.pt" for i in range(3)]
    reset_rng(100)
    data, metadata = _gen_data(return_metadata=True)

    required = [prox_path, fedavg_path, central_path, *local_paths,
                *ensemble_paths]
    if not all(path.exists() for path in required):
        print("[models] training Prox-FL", flush=True)
        prox, _, _ = train_federated(ProxFLClient, data, seed=666)
        save_model(prox_path, prox)
        print("[models] training FedAvg", flush=True)
        fedavg, _, _ = train_federated(FedAvgClient, data, seed=666)
        save_model(fedavg_path, fedavg)
        print("[models] training vehicle-local predictors", flush=True)
        local = train_local(data, seed=666)
        for path, model in zip(local_paths, local):
            save_model(path, model)
        print("[models] training centralized reference", flush=True)
        central = train_centralized(data, seed=666, epochs=90)
        save_model(central_path, central)
        for index, path in enumerate(ensemble_paths):
            print(f"[models] training ensemble member {index + 1}/3", flush=True)
            save_model(path, train_centralized(data, seed=900 + index,
                                               epochs=70))
        metadata_path.write_text(json.dumps({
            "training_clients": metadata,
            "rounds": Config.ROUNDS,
            "local_epochs": Config.LOCAL_EPOCHS,
            "samples_per_client": Config.N_DATA,
            "centralized_epochs": 90,
            "ensemble_members": 3,
        }, indent=2), encoding="utf-8")

    prox = load_model(prox_path)
    fedavg = load_model(fedavg_path)
    central = load_model(central_path)
    local = [load_model(path) for path in local_paths]
    ensemble = EnsemblePredictor([load_model(path) for path in ensemble_paths])
    return data, metadata, prox, fedavg, central, local, ensemble


def run_panel(name: str, methods: dict, seeds, *, vehicle_masses=None,
              vehicle_drags=None, noise=0.0, disturbance=0.0):
    run_path = OUT / f"{name}_runs.csv"
    route_path = OUT / f"{name}_routes.csv"
    done = completed_keys(run_path, ["Method", "Seed"])
    for method, factory in methods.items():
        for seed in seeds:
            key = (method, seed)
            if key in done:
                continue
            print(f"[{name}] {method} | seed={seed}", flush=True)
            env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
            result = run_simulation(
                factory(env), env, seed=seed, noise=noise,
                disturbance=disturbance, vehicle_masses=vehicle_masses,
                vehicle_drags=vehicle_drags)
            scalar = {k: v for k, v in result.items()
                      if isinstance(v, (int, float, np.number))}
            append_row(run_path, {"Method": method, "Seed": seed, **scalar})
            for route in result["RouteMetrics"]:
                append_row(route_path, {"Method": method, "Seed": seed,
                                        **route})
    return pd.read_csv(run_path), pd.read_csv(route_path)


def cluster_bootstrap(values, seed=20261103, n_boot=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot = values[draws].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def summarize_runs(frame: pd.DataFrame, prefix: str):
    rows = []
    for method, sub in frame.groupby("Method", sort=False):
        row = {"Method": method, "Seeds": len(sub)}
        for metric in ["SR", "SC", "CR", "AvgDist", "MinClearance",
                       "MeanViolationDepth", "MaxViolationDepth",
                       "MeanViolationDuration", "MaxViolationDuration",
                       "Cost", "Jerk", "ComputeMs", "ComputeP95Ms"]:
            values = sub[metric].to_numpy(float)
            lo, hi = cluster_bootstrap(values)
            scale = 100.0 if metric in {"SR", "CR"} else 1.0
            row[f"{metric}_mean"] = scale * float(values.mean())
            row[f"{metric}_sd"] = scale * float(values.std(ddof=1))
            row[f"{metric}_ci_lo"] = scale * lo
            row[f"{metric}_ci_hi"] = scale * hi
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"{prefix}_summary.csv", index=False)
    return result


def paired_contrasts(frame: pd.DataFrame, routes: pd.DataFrame,
                     reference="FedUMPC", prefix="main"):
    rows = []
    ref = frame[frame.Method == reference].set_index("Seed")
    ref_routes = routes[routes.Method == reference].copy()
    for method in frame.Method.drop_duplicates():
        if method == reference:
            continue
        cmp = frame[frame.Method == method].set_index("Seed")
        common = ref.index.intersection(cmp.index)
        row = {"Reference": reference, "Comparator": method,
               "Paired_seeds": len(common)}
        for metric in ["SR", "SC", "AvgDist", "MinClearance",
                       "MeanViolationDepth", "ComputeMs"]:
            diff = (ref.loc[common, metric] - cmp.loc[common, metric]).to_numpy(float)
            lo, hi = cluster_bootstrap(diff)
            scale = 100.0 if metric == "SR" else 1.0
            row[f"Delta_{metric}"] = scale * float(diff.mean())
            row[f"Delta_{metric}_ci_lo"] = scale * lo
            row[f"Delta_{metric}_ci_hi"] = scale * hi

        cmp_routes = routes[routes.Method == method].copy()
        merged = ref_routes.merge(cmp_routes, on=["Seed", "vehicle"],
                                  suffixes=("_ref", "_cmp"))
        discordant_ref = int(((merged.success_ref == 1) &
                              (merged.success_cmp == 0)).sum())
        discordant_cmp = int(((merged.success_ref == 0) &
                              (merged.success_cmp == 1)).sum())
        discordant = discordant_ref + discordant_cmp
        row["Success_discordant_ref_wins"] = discordant_ref
        row["Success_discordant_cmp_wins"] = discordant_cmp
        row["McNemar_exact_p"] = (1.0 if discordant == 0 else
                                   float(binomtest(discordant_ref,
                                                 discordant, 0.5).pvalue))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(OUT / f"{prefix}_paired_contrasts.csv", index=False)
    return result


def primary_and_mismatch(prox, central, local, ensemble, fedavg=None):
    methods = {
        "Linear MPC": lambda env: [LinearMPC(env) for _ in range(4)],
        "Tube-style MPC": lambda env: [TubeMPC(env) for _ in range(4)],
        "Scenario MPC": lambda env: [StochasticMPC(env) for _ in range(4)],
        "GP-margin MPC": lambda env: [GP_MPC(env) for _ in range(4)],
        "Fixed-margin MPC": lambda env: [RobustMPC(env) for _ in range(4)],
        "Ensemble UMPC": lambda env: [FedUMPCController(env, ensemble)
                                       for _ in range(4)],
        "Centralized UMPC": lambda env: [FedUMPCController(env, central)
                                          for _ in range(4)],
        "FedUMPC": lambda env: [FedUMPCController(env, prox)
                                 for _ in range(4)],
    }
    runs, routes = run_panel("r1_main", methods, TEST_SEEDS)
    summary = summarize_runs(runs, "r1_main")
    paired_contrasts(runs, routes, prefix="r1_main")

    mismatch_methods = {
        "Local UMPC": lambda env: [FedUMPCController(env, local[i])
                                    for i in range(4)],
        "FedAvg UMPC": lambda env: [FedUMPCController(env, fedavg)
                                      for _ in range(4)],
        "GP-margin MPC": methods["GP-margin MPC"],
        "Fixed-margin MPC": methods["Fixed-margin MPC"],
        "Ensemble UMPC": methods["Ensemble UMPC"],
        "Centralized UMPC": methods["Centralized UMPC"],
        "FedUMPC": methods["FedUMPC"],
    }
    masses = [650.0, 850.0, 1150.0, 1350.0]
    drags = [0.03, 0.05, 0.08, 0.11]
    mm_runs, mm_routes = run_panel(
        "r1_mismatch", mismatch_methods, TEST_SEEDS,
        vehicle_masses=masses, vehicle_drags=drags)
    summarize_runs(mm_runs, "r1_mismatch")
    paired_contrasts(mm_runs, mm_routes, prefix="r1_mismatch")
    return summary


def heldout_uncertainty(model, metadata, n_per_client=500, mc_samples=10):
    rng = np.random.default_rng(20261103)
    rows = []
    scenarios = {
        "Nominal": (0.0, 0.0, [1.0] * 4),
        "High noise": (0.25, 0.0, [1.0] * 4),
        "Wind": (0.0, 1.0, [1.0] * 4),
        "Mass shift": (0.0, 0.0, [1.5, 0.65, 1.3, 0.75]),
    }
    for scenario, (noise, disturbance, scales) in scenarios.items():
        states, actions, targets, clients = [], [], [], []
        for i in range(4):
            plant = VehicleModel(metadata[i]["mass_kg"] * scales[i],
                                 metadata[i]["drag"])
            for _ in range(n_per_client):
                state = np.array([rng.uniform(-22, 22), rng.uniform(-22, 22),
                                  rng.uniform(0, 12),
                                  rng.uniform(-np.pi, np.pi)])
                action = rng.uniform([-Config.MAX_STEER, -Config.MAX_ACCEL],
                                     [Config.MAX_STEER, Config.MAX_ACCEL])
                np.random.seed(int(rng.integers(0, 2**31 - 1)))
                next_state = plant.step(state, action, noise_std=noise,
                                        disturbance=disturbance)
                states.append(state); actions.append(action)
                targets.append(next_state - state); clients.append(i + 1)
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        targets = np.asarray(targets, dtype=np.float32)
        means, variances = [], []
        for start in range(0, len(states), 500):
            mean, var = model.predict_horizon(
                states[start:start + 500], actions[start:start + 500],
                num_samples=mc_samples)
            means.append(mean); variances.append(var)
        mean = np.vstack(means)
        uncertainty = np.sqrt(np.maximum(np.concatenate(variances), 0.0))
        # Match the dimensionless uncertainty definition: errors are measured
        # after scaling each state-increment component by the BNN output scale.
        output_scale = model.delta_scale.detach().cpu().numpy()
        error = np.linalg.norm((mean - targets) / output_scale, axis=1)
        for client, unc, err in zip(clients, uncertainty, error):
            rows.append({"Scenario": scenario, "Client": client,
                         "M": mc_samples, "Uncertainty": float(unc),
                         "Prediction_error": float(err)})
    return pd.DataFrame(rows)


def uncertainty_diagnostics(prox, metadata):
    path = OUT / "r1_uncertainty_samples.csv"
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frame = heldout_uncertainty(prox, metadata, n_per_client=500,
                                    mc_samples=10)
        frame.to_csv(path, index=False)
    q_u = frame.Uncertainty.quantile(0.25)
    q_e = frame.Prediction_error.quantile(0.75)
    frame["False_low_uncertainty"] = (
        (frame.Uncertainty <= q_u) & (frame.Prediction_error >= q_e))
    bins = pd.qcut(frame.Uncertainty, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    conditional = (frame.assign(Uncertainty_quartile=bins)
                    .groupby(["Scenario", "Uncertainty_quartile"],
                             observed=True)
                    .Prediction_error
                    .agg(["count", "mean", "median",
                          lambda x: np.quantile(x, 0.90),
                          lambda x: np.quantile(x, 0.95)])
                    .reset_index())
    conditional.columns = ["Scenario", "Uncertainty_quartile", "N",
                           "Mean_error", "Median_error", "P90_error",
                           "P95_error"]
    conditional.to_csv(OUT / "r1_uncertainty_conditional.csv", index=False)
    rows = []
    for scenario, sub in list(frame.groupby("Scenario")) + [("Pooled", frame)]:
        rho, p = spearmanr(sub.Uncertainty, sub.Prediction_error)
        rows.append({
            "Scenario": scenario, "N": len(sub), "Spearman_rho": rho,
            "p_value": p,
            "False_low_uncertainty_pct": 100 * sub.False_low_uncertainty.mean(),
            "Mean_uncertainty": sub.Uncertainty.mean(),
            "Mean_error": sub.Prediction_error.mean(),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "r1_uncertainty_summary.csv", index=False)
    return frame, summary


def mechanism_isolation(prox, uncertainty_values):
    # Match the mean clipped margin at every horizon index over the fixed
    # held-out query set, rather than clipping the mean U (Jensen mismatch).
    calibration_u = np.asarray(uncertainty_values).reshape(-1, 1)
    fixed_profile = (Config.SAFETY_MARGIN + np.mean(np.clip(
        Config.ROBUST_BETA * Config.UNCERTAINTY_MARGIN_SCALE
        * calibration_u * (1 + 0.15 * np.arange(Config.HORIZON))[None, :],
        0.0, Config.MAX_UNCERTAINTY_MARGIN), axis=0))

    class FixedGrowthMPC(LinearMPC):
        """Non-learned reference with a frozen, mean-matched margin profile."""
        def _cost(self, u_flat, state, target):
            u = u_flat.reshape(self.H, 2)
            states = self._rollout(u, state)
            return self._base_cost(states, u, target) + sum(
                self._obstacle_cost(states[h], fixed_profile[h])
                for h in range(self.H))
    methods = {
        "Full FedUMPC": lambda env: [FedUMPCController(env, prox)
                                      for _ in range(4)],
        "No uncertainty response": lambda env: [
            FedUMPCController(env, prox, False) for _ in range(4)],
        "Risk only": lambda env: [FedUMPCController(
            env, prox, use_risk_cost=True, use_model_trust=False,
            use_adaptive_margin=False) for _ in range(4)],
        "Trust only": lambda env: [FedUMPCController(
            env, prox, use_risk_cost=False, use_model_trust=True,
            use_adaptive_margin=False) for _ in range(4)],
        "Margin only": lambda env: [FedUMPCController(
            env, prox, use_risk_cost=False, use_model_trust=False,
            use_adaptive_margin=True) for _ in range(4)],
        "Risk + trust": lambda env: [FedUMPCController(
            env, prox, use_adaptive_margin=False) for _ in range(4)],
        "Risk + margin": lambda env: [FedUMPCController(
            env, prox, use_model_trust=False) for _ in range(4)],
        "Trust + margin": lambda env: [FedUMPCController(
            env, prox, use_risk_cost=False) for _ in range(4)],
        "No candidate screen": lambda env: [FedUMPCController(
            env, prox, use_candidate_screen=False) for _ in range(4)],
        "Constant U": lambda env: [FedUMPCController(
            env, prox, uncertainty_mode="constant") for _ in range(4)],
        "Shuffled U": lambda env: [FedUMPCController(
            env, prox, uncertainty_mode="shuffled") for _ in range(4)],
        "Fixed-growth MPC": lambda env: [FixedGrowthMPC(env) for _ in range(4)],
    }
    runs, routes = run_panel("r1_mechanisms", methods, TEST_SEEDS)
    summary = summarize_runs(runs, "r1_mechanisms")
    paired_contrasts(runs, routes, reference="Full FedUMPC",
                     prefix="r1_mechanisms")
    pd.DataFrame({"h": np.arange(Config.HORIZON),
                  "fixed_margin_m": fixed_profile}).to_csv(
        OUT / "r1_fixed_growth_margin.csv", index=False)
    return summary


def mc_sample_sweep(prox, metadata):
    query_path = OUT / "r1_mc_query_diagnostics.csv"
    if not query_path.exists():
        rng = np.random.default_rng(20261104)
        states = np.c_[rng.uniform(-22, 22, 256),
                       rng.uniform(-22, 22, 256),
                       rng.uniform(0, 12, 256),
                       rng.uniform(-np.pi, np.pi, 256)].astype(np.float32)
        actions = rng.uniform([-Config.MAX_STEER, -Config.MAX_ACCEL],
                              [Config.MAX_STEER, Config.MAX_ACCEL],
                              size=(256, 2)).astype(np.float32)
        for M in [5, 10, 20, 40]:
            # Exclude one-time framework and kernel initialization from the
            # repeated-query timing reported for each sampling level.
            prox.predict_horizon(states[:8], actions[:8], num_samples=M)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            values, timings = [], []
            for repeat in range(20):
                reset_rng(8000 + repeat)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                _, var = prox.predict_horizon(states, actions, num_samples=M)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings.append(1000 * (time.perf_counter() - start))
                values.append(np.sqrt(np.maximum(var, 0.0)))
            values = np.asarray(values)
            repeat_var = values.var(axis=0, ddof=1)
            append_row(query_path, {
                "M": M, "Batch_queries": len(states),
                "Mean_batch_latency_ms": np.mean(timings),
                "P95_batch_latency_ms": np.percentile(timings, 95),
                "Mean_U": values.mean(),
                "Mean_query_repeat_variance": repeat_var.mean(),
                "P90_query_repeat_variance": np.percentile(repeat_var, 90),
            })

    rows = []
    for M in [5, 10, 20, 40]:
        path = OUT / f"r1_mc_validation_M{M}.csv"
        if path.exists():
            frame = pd.read_csv(path)
        else:
            frame = heldout_uncertainty(prox, metadata, n_per_client=150,
                                        mc_samples=M)
            frame.to_csv(path, index=False)
        rho, p = spearmanr(frame.Uncertainty, frame.Prediction_error)
        rows.append({"M": M, "N": len(frame), "Spearman_rho": rho,
                     "p_value": p, "Mean_U": frame.Uncertainty.mean(),
                     "Mean_error": frame.Prediction_error.mean()})
    pd.DataFrame(rows).to_csv(OUT / "r1_mc_validation_summary.csv",
                              index=False)

    run_path = OUT / "r1_mc_closed_loop_runs.csv"
    route_path = OUT / "r1_mc_closed_loop_routes.csv"
    done = completed_keys(run_path, ["M", "Seed"])
    original_M = prox._M
    try:
        for M in [5, 10, 20, 40]:
            prox._M = M
            for seed in MC_TEST_SEEDS:
                if (M, seed) in done:
                    continue
                print(f"[MC sweep] M={M} | seed={seed}", flush=True)
                env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
                result = run_simulation(
                    [FedUMPCController(env, prox) for _ in range(4)],
                    env, seed=seed)
                scalar = {k: v for k, v in result.items()
                          if isinstance(v, (int, float, np.number))}
                append_row(run_path, {"M": M, "Seed": seed, **scalar})
                for route in result["RouteMetrics"]:
                    append_row(route_path, {"M": M, "Seed": seed, **route})
    finally:
        prox._M = original_M
    runs = pd.read_csv(run_path)
    summary_rows = []
    for M, sub in runs.groupby("M"):
        summary_rows.append({
            "M": M, "Seeds": len(sub),
            "SR_pct": 100 * sub.SR.mean(),
            "SC_pct": sub.SC.mean(),
            "Mean_controller_ms": sub.ComputeMs.mean(),
            "Mean_run_p9_ms": sub.ComputeP9Ms.mean(),
            "Mean_run_p90_ms": sub.ComputeP90Ms.mean(),
            "Mean_run_p95_ms": sub.ComputeP95Ms.mean(),
            "Mean_violation_depth_m": sub.MeanViolationDepth.mean(),
        })
    pd.DataFrame(summary_rows).to_csv(
        OUT / "r1_mc_closed_loop_summary.csv", index=False)


def candidate_scaling(prox):
    path = OUT / "r1_candidate_latency.csv"
    if path.exists():
        return pd.read_csv(path)
    rng = np.random.default_rng(20261105)
    env = CrossingEnv(seed=42, obstacle_density=D_OBS)
    controller = FedUMPCController(env, prox)
    state = np.array([-20.0, -2.0, 3.0, 0.0])
    target = np.array([20.0, -2.0, 3.0, 0.0])
    original_M = prox._M
    for M in [5, 10, 20, 40]:
        prox._M = M
        for count in [4, 8, 12, 16]:
            guesses = rng.uniform(controller._lo, controller._hi,
                                  size=(count, len(controller._lo)))
            controller._cost_many(guesses, state, target)
            timings = []
            for _ in range(30):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                controller._cost_many(guesses, state, target)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings.append(1000 * (time.perf_counter() - start))
            append_row(path, {"M": M, "Candidates": count,
                              "Mean_ms": np.mean(timings),
                              "P9_ms": np.percentile(timings, 9),
                              "P90_ms": np.percentile(timings, 90),
                              "P95_ms": np.percentile(timings, 95)})
    prox._M = original_M
    return pd.read_csv(path)


def generate_fleet_data(n_clients, seed, sparse_fraction=0.25):
    rng = np.random.default_rng(seed)
    data, holdout, metadata = {}, {}, {}
    for i in range(n_clients):
        mass = 650.0 + 700.0 * i / max(1, n_clients - 1)
        drag = 0.02 + 0.10 * ((i * 7) % max(1, n_clients)) / max(1, n_clients - 1)
        phase = 2 * np.pi * i / n_clients
        plant = VehicleModel(mass=mass, drag=drag)
        n_train = 25 if i < max(1, int(sparse_fraction * n_clients)) else 120
        client_train, client_test = [], []
        for count, target in [(n_train, client_train), (180, client_test)]:
            for _ in range(count):
                radial = rng.uniform(2.0, 22.0)
                angle = phase + rng.normal(0.0, 0.35)
                state = np.array([radial * np.cos(angle), radial * np.sin(angle),
                                  rng.uniform(0, 12),
                                  rng.uniform(-np.pi, np.pi)])
                action = rng.uniform([-Config.MAX_STEER, -Config.MAX_ACCEL],
                                     [Config.MAX_STEER, Config.MAX_ACCEL])
                next_state = plant.step(state, action)
                target.append((state, action, next_state))
        data[i] = client_train
        holdout[i] = client_test
        metadata[i] = {"mass": mass, "drag": drag, "n_train": n_train,
                       "sparse": i < max(1, int(sparse_fraction * n_clients))}
    return data, holdout, metadata


def empirical_dissimilarity(model, data, max_samples=64):
    gradients = []
    for samples in data.values():
        chosen = samples[:max_samples]
        states = np.asarray([s for s, _, _ in chosen], dtype=np.float32)
        actions = np.asarray([u for _, u, _ in chosen], dtype=np.float32)
        targets = np.asarray([sn - s for s, _, sn in chosen], dtype=np.float32)
        xu = torch.as_tensor(np.c_[states, actions], dtype=torch.float32,
                             device=Config.DEVICE)
        y = torch.as_tensor(targets, dtype=torch.float32,
                            device=Config.DEVICE)
        model.zero_grad(set_to_none=True)
        loss = torch.mean((model(xu) - y) ** 2)
        loss.backward()
        gradients.append(torch.cat([p.grad.detach().flatten()
                                    for p in model.parameters()]))
    stack = torch.stack(gradients)
    numerator = torch.sqrt(torch.mean(torch.sum(stack ** 2, dim=1)))
    denominator = torch.linalg.vector_norm(torch.mean(stack, dim=0)) + 1e-12
    return float((numerator / denominator).item())


def evaluate_predictors(local_models, global_models, holdout, metadata):
    rows = []
    for client_id, samples in holdout.items():
        if local_models:
            rows.append({"Method": "Local Only", "Client": client_id,
                         "Sparse": metadata[client_id]["sparse"],
                         "MSE": mse_model(local_models[client_id], samples)})
        for name, model in global_models.items():
            rows.append({"Method": name, "Client": client_id,
                         "Sparse": metadata[client_id]["sparse"],
                         "MSE": mse_model(model, samples)})
    return pd.DataFrame(rows)


def fleet_scaling():
    path = OUT / "r1_fleet_scaling.csv"
    if not path.exists():
        for n_clients in [4, 16, 32]:
            for seed in FEDERATION_SEEDS:
                print(f"[fleet scaling] N={n_clients} | seed={seed}", flush=True)
                data, holdout, metadata = generate_fleet_data(n_clients, seed)
                with temporary_config(ROUNDS=12, LOCAL_EPOCHS=3,
                                      BATCH_SIZE=32, LR=0.004):
                    local = train_local(data, seed=seed, rounds=12)
                    fedavg, _, fedavg_drift = train_federated(
                        FedAvgClient, data, seed=seed, rounds=12)
                    prox, _, prox_drift = train_federated(
                        ProxFLClient, data, seed=seed, rounds=12)
                    central = train_centralized(data, seed=seed, epochs=45)
                detail = evaluate_predictors(
                    local, {"FedAvg": fedavg, "Prox-FL": prox,
                            "Centralized": central}, holdout, metadata)
                detail.to_csv(OUT / f"r1_fleet_detail_N{n_clients}_S{seed}.csv",
                              index=False)
                for method, sub in detail.groupby("Method"):
                    model = {"FedAvg": fedavg, "Prox-FL": prox,
                             "Centralized": central}.get(method)
                    drift = ({"FedAvg": fedavg_drift,
                              "Prox-FL": prox_drift}.get(method, np.nan))
                    append_row(path, {
                        "N": n_clients, "Seed": seed, "Method": method,
                        "Mean_MSE": sub.MSE.mean(),
                        "Sparse_client_MSE": sub[sub.Sparse].MSE.mean(),
                        "Rich_client_MSE": sub[~sub.Sparse].MSE.mean(),
                        "Final_drift": drift,
                        "Empirical_B": (empirical_dissimilarity(model, data)
                                        if model is not None else np.nan),
                        "Model_payload_MB": (model_bytes(model) / 1e6
                                             if model is not None else np.nan),
                        "Aggregate_round_traffic_MB": (
                            2 * n_clients * model_bytes(model) / 1e6
                            if model is not None else np.nan),
                    })
    frame = pd.read_csv(path)
    (frame.groupby(["N", "Method"], as_index=False)
          .agg(Mean_MSE=("Mean_MSE", "mean"),
               Mean_MSE_sd=("Mean_MSE", "std"),
               Sparse_client_MSE=("Sparse_client_MSE", "mean"),
               Sparse_client_MSE_sd=("Sparse_client_MSE", "std"),
               Final_drift=("Final_drift", "mean"),
               Empirical_B=("Empirical_B", "mean"),
               Aggregate_round_traffic_MB=("Aggregate_round_traffic_MB", "mean"))
          .to_csv(OUT / "r1_fleet_scaling_summary.csv", index=False))

    stress_path = OUT / "r1_communication_stress.csv"
    if not stress_path.exists():
        for seed in FEDERATION_SEEDS:
            data, holdout, metadata = generate_fleet_data(16, seed)
            for label, participation, staleness in [
                ("Synchronous full", 1.0, 0),
                ("50% participation", 0.5, 0),
                ("50% + two-round stale", 0.5, 2),
            ]:
                print(f"[communication stress] {label} | seed={seed}",
                      flush=True)
                with temporary_config(ROUNDS=12, LOCAL_EPOCHS=3,
                                      BATCH_SIZE=32, LR=0.004):
                    model, _, drift = train_federated(
                        ProxFLClient, data, seed=seed, rounds=12,
                        participation=participation, staleness=staleness)
                detail = evaluate_predictors([], {"Prox-FL": model}, holdout,
                                             metadata)
                sub = detail[detail.Method == "Prox-FL"]
                append_row(stress_path, {
                    "Scenario": label, "Seed": seed,
                    "Participation": participation, "Staleness_rounds": staleness,
                    "Mean_MSE": sub.MSE.mean(),
                    "Sparse_client_MSE": sub[sub.Sparse].MSE.mean(),
                    "Final_drift": drift,
                    "Round_traffic_MB": (2 * 16 * participation
                                         * model_bytes(model) / 1e6),
                })
        stress = pd.read_csv(stress_path)
        (stress.groupby("Scenario", as_index=False)
               .agg(Mean_MSE=("Mean_MSE", "mean"),
                    Mean_MSE_sd=("Mean_MSE", "std"),
                    Sparse_client_MSE=("Sparse_client_MSE", "mean"),
                    Final_drift=("Final_drift", "mean"),
                    Round_traffic_MB=("Round_traffic_MB", "mean"))
               .to_csv(OUT / "r1_communication_stress_summary.csv", index=False))


def obstacle_sensitivity(prox):
    path = OUT / "r1_obstacle_sensitivity.csv"
    settings = []
    for name, base, values in [
        ("q_obs", 7800.0, [5460.0, 7800.0, 10140.0]),
        ("c_s", 0.02, [0.01, 0.02, 0.04]),
        ("inner_a", 1.15, [0.80, 1.15, 1.50]),
        ("d_inf", 4.2, [3.8, 4.2, 4.8]),
    ]:
        for value in values:
            settings.append((name, base, value))
    done = completed_keys(path, ["Parameter", "Value", "Seed"])
    for parameter, base, value in settings:
        for seed in SENSITIVITY_SEEDS:
            if (parameter, value, seed) in done:
                continue
            print(f"[obstacle sensitivity] {parameter}={value} | seed={seed}",
                  flush=True)
            updates = {
                "OBSTACLE_PENALTY": value if parameter == "q_obs" else 7800.0,
                "FED_SOFT_OBS_WEIGHT": value if parameter == "c_s" else 0.02,
                "FED_INNER_OBS_WEIGHT": value if parameter == "inner_a" else 1.15,
                "FED_OBSTACLE_INFLUENCE": value if parameter == "d_inf" else 4.2,
            }
            with temporary_config(**updates):
                env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
                result = run_simulation(
                    [FedUMPCController(env, prox) for _ in range(4)],
                    env, seed=seed)
            scalar = {k: v for k, v in result.items()
                      if isinstance(v, (int, float, np.number))}
            append_row(path, {"Parameter": parameter, "Base": base,
                              "Value": value, "Seed": seed, **scalar})
    frame = pd.read_csv(path)
    (frame.groupby(["Parameter", "Base", "Value"], as_index=False)
          .agg(SR=("SR", "mean"), SC=("SC", "mean"),
               MinClearance=("MinClearance", "mean"),
               MeanViolationDepth=("MeanViolationDepth", "mean"),
               Cost=("Cost", "mean"))
          .to_csv(OUT / "r1_obstacle_sensitivity_summary.csv", index=False))


def communication_budget(prox):
    payload_mb = model_bytes(prox) / 1e6
    rows = []
    for n_clients in [4, 16, 32]:
        for rate_mbps in [5, 20, 100]:
            one_way_ms = 1000 * (8 * payload_mb / rate_mbps)
            rows.append({
                "Clients": n_clients, "Link_rate_Mbps": rate_mbps,
                "Model_payload_MB": payload_mb,
                "Per_client_round_trip_payload_MB": 2 * payload_mb,
                "Per_client_serialized_transfer_ms": 2 * one_way_ms,
                "Aggregate_round_traffic_MB": 2 * n_clients * payload_mb,
                "Scope": "payload-only; excludes protocol, queueing, and RTT",
            })
    pd.DataFrame(rows).to_csv(OUT / "r1_communication_budget.csv", index=False)


def panel_label(ax, label):
    ax.text(-0.13, 1.04, label, transform=ax.transAxes, ha="left",
            va="bottom", fontsize=8.5, fontweight="bold", clip_on=False)


def method_colors(methods):
    colors = []
    for index, method in enumerate(methods):
        colors.append(FED_RED if "FedUMPC" in method else PALETTE[index % len(PALETTE)])
    return colors


def plot_main_figure():
    main = pd.read_csv(OUT / "r1_main_summary.csv")
    mismatch = pd.read_csv(OUT / "r1_mismatch_summary.csv")
    contrasts = pd.read_csv(OUT / "r1_main_paired_contrasts.csv")
    methods = list(main.Method)
    colors = method_colors(methods)
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 5.4))
    x = np.arange(len(methods))
    for ax, metric, ylabel in [
        (axes[0, 0], "SR", "Success rate (%)"),
        (axes[0, 1], "SC", "Safety compliance (%)"),
        (axes[0, 2], "MinClearance", "Minimum clearance (m)"),
    ]:
        y = main[f"{metric}_mean"].to_numpy(float)
        lo = main[f"{metric}_ci_lo"].to_numpy(float)
        hi = main[f"{metric}_ci_hi"].to_numpy(float)
        ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none",
                    ecolor="#333333", capsize=2.2, lw=0.8, zorder=3)
        ax.scatter(x, y, c=colors, s=24, edgecolor="white", linewidth=0.5,
                   zorder=4)
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=72, ha="right")
        ax.set_ylabel(ylabel); ax.grid(axis="y", ls=":", alpha=0.45)
    mm_methods = list(mismatch.Method)
    mm_colors = method_colors(mm_methods)
    axes[1, 0].scatter(mismatch.SR_mean, mismatch.SC_mean,
                       s=45 + 30 * (1 - mismatch.CR_mean / 100.0),
                       c=mm_colors, edgecolor="white", linewidth=0.5)
    for _, row in mismatch.iterrows():
        axes[1, 0].annotate(row.Method.replace(" MPC", ""),
                            (row.SR_mean, row.SC_mean), xytext=(3, 2),
                            textcoords="offset points", fontsize=6.2)
    axes[1, 0].set_xlabel("Mismatched-plant success rate (%)")
    axes[1, 0].set_ylabel("Mismatched-plant compliance (%)")
    axes[1, 0].grid(ls=":", alpha=0.45)

    show = contrasts.sort_values("Delta_SC", ascending=True)
    y = np.arange(len(show))
    axes[1, 1].errorbar(show.Delta_SC, y,
                        xerr=[show.Delta_SC - show.Delta_SC_ci_lo,
                              show.Delta_SC_ci_hi - show.Delta_SC],
                        fmt="o", color=FED_RED, ecolor="#4D4D4D",
                        capsize=2.2)
    axes[1, 1].axvline(0, color="#4D4D4D", lw=0.8, ls="--")
    axes[1, 1].set_yticks(y); axes[1, 1].set_yticklabels(show.Comparator)
    axes[1, 1].set_xlabel("Paired SC difference (pp)\nFedUMPC - comparator")
    axes[1, 1].grid(axis="x", ls=":", alpha=0.45)

    safety = main[["Method", "MeanViolationDepth_mean",
                   "MaxViolationDuration_mean"]].copy()
    sx = np.arange(len(safety))
    axes[1, 2].bar(sx, safety.MeanViolationDepth_mean, color=colors,
                   width=0.68)
    ax2 = axes[1, 2].twinx()
    ax2.plot(sx, safety.MaxViolationDuration_mean, color="#4D4D4D",
             marker="o", ls="--")
    axes[1, 2].set_xticks(sx)
    axes[1, 2].set_xticklabels(safety.Method, rotation=72, ha="right")
    axes[1, 2].set_ylabel("Mean violation depth (m)")
    ax2.set_ylabel("Max duration (steps)")
    axes[1, 2].grid(axis="y", ls=":", alpha=0.45)
    for ax, label in zip(axes.ravel(), "abcdef"):
        panel_label(ax, f"({label})")
    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.15, top=0.98,
                        wspace=0.48, hspace=0.52)
    fig.savefig(OUT / "r1_main_benchmark.pdf")
    plt.close(fig)


def plot_mechanism_figure():
    summary = pd.read_csv(OUT / "r1_mechanisms_summary.csv")
    uncertainty = pd.read_csv(OUT / "r1_uncertainty_samples.csv")
    mc = pd.read_csv(OUT / "r1_mc_closed_loop_summary.csv")
    query = pd.read_csv(OUT / "r1_mc_query_diagnostics.csv")
    methods = list(summary.Method)
    colors = method_colors(methods)
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 5.15))
    x = np.arange(len(methods))
    for ax, metric, ylabel in [
        (axes[0, 0], "SR", "Success rate (%)"),
        (axes[0, 1], "SC", "Safety compliance (%)"),
    ]:
        values = summary[f"{metric}_mean"]
        ax.bar(x, values, color=colors, width=0.72)
        ax.set_xticks(x); ax.set_xticklabels(methods, rotation=70, ha="right")
        ax.set_ylabel(ylabel); ax.grid(axis="y", ls=":", alpha=0.45)
    bins = pd.qcut(uncertainty.Uncertainty, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    values = [uncertainty.loc[bins == label, "Prediction_error"].to_numpy()
              for label in ["Q1", "Q2", "Q3", "Q4"]]
    violin = axes[0, 2].violinplot(values, showmeans=True, showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor(FED_RED); body.set_alpha(0.6)
    axes[0, 2].set_xticks([1, 2, 3, 4]); axes[0, 2].set_xticklabels(["Q1", "Q2", "Q3", "Q4"])
    axes[0, 2].set_xlabel("Uncertainty quartile")
    axes[0, 2].set_ylabel("Normalized one-step error (-)")
    axes[0, 2].grid(axis="y", ls=":", alpha=0.45)

    sample = uncertainty.sample(min(1200, len(uncertainty)), random_state=2)
    axes[1, 0].scatter(sample.Uncertainty, sample.Prediction_error, s=5,
                       alpha=0.24, color=FED_RED, edgecolors="none")
    q_u = uncertainty.Uncertainty.quantile(0.25)
    q_e = uncertainty.Prediction_error.quantile(0.75)
    axes[1, 0].axvline(q_u, color="#007FFF", ls="--", lw=0.9)
    axes[1, 0].axhline(q_e, color="#4D4D4D", ls=":", lw=0.9)
    axes[1, 0].set_xlabel("Normalized MC-dropout uncertainty (-)")
    axes[1, 0].set_ylabel("Normalized one-step error (-)")
    axes[1, 0].grid(ls=":", alpha=0.3)

    axes[1, 1].plot(mc.M, mc.SR_pct, marker="o", color=FED_RED,
                    label="SR (%)")
    axes[1, 1].plot(mc.M, mc.SC_pct, marker="s", color=PALETTE[1],
                    label="SC (%)")
    axes[1, 1].set_xlabel("MC-dropout samples M")
    axes[1, 1].set_ylabel("Closed-loop endpoint (%)")
    axes[1, 1].grid(ls=":", alpha=0.45)
    axes[1, 1].legend(ncol=2, frameon=False)

    axes[1, 2].plot(query.M, query.Mean_batch_latency_ms, marker="o",
                    color=FED_RED, label="Latency")
    ax2 = axes[1, 2].twinx()
    ax2.plot(query.M, query.Mean_query_repeat_variance, marker="s",
             color="#007FFF", label="Repeat variance")
    axes[1, 2].set_xlabel("MC-dropout samples M")
    axes[1, 2].set_ylabel("256-query latency (ms)")
    ax2.set_ylabel("Mean repeat variance")
    axes[1, 2].grid(ls=":", alpha=0.45)
    for ax, label in zip(axes.ravel(), "abcdef"):
        panel_label(ax, f"({label})")
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.14, top=0.98,
                        wspace=0.48, hspace=0.50)
    fig.savefig(OUT / "r1_mechanism_validation.pdf")
    plt.close(fig)


def plot_federation_figure():
    frame = pd.read_csv(OUT / "r1_fleet_scaling_summary.csv")
    stress = pd.read_csv(OUT / "r1_communication_stress_summary.csv")
    methods = ["Local Only", "FedAvg", "Prox-FL", "Centralized"]
    colors = {m: (FED_RED if m == "Prox-FL" else PALETTE[i])
              for i, m in enumerate(methods)}
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 4.55))
    for method in methods:
        sub = frame[frame.Method == method]
        axes[0, 0].plot(sub.N, sub.Sparse_client_MSE, marker="o",
                        color=colors[method], label=method)
        axes[0, 1].plot(sub.N, sub.Mean_MSE, marker="o",
                        color=colors[method], label=method)
    axes[0, 0].set_ylabel("Data-poor normalized MSE (-)")
    axes[0, 1].set_ylabel("Fleet-wide normalized MSE (-)")
    for ax in axes[0]:
        ax.set_xlabel("Fleet size N"); ax.set_xticks([4, 16, 32])
        ax.grid(ls=":", alpha=0.45)
    axes[0, 0].legend(ncol=2, frameon=False)

    prox = frame[frame.Method == "Prox-FL"]
    axes[1, 0].plot(prox.N, prox.Final_drift, marker="o", color=FED_RED,
                    label="Client drift")
    ax2 = axes[1, 0].twinx()
    ax2.plot(prox.N, prox.Empirical_B, marker="s", color="#007FFF",
             label=r"Empirical $B$")
    axes[1, 0].set_xlabel("Fleet size N")
    axes[1, 0].set_ylabel("Final client drift")
    ax2.set_ylabel(r"Empirical dissimilarity $B$")
    axes[1, 0].set_xticks([4, 16, 32]); axes[1, 0].grid(ls=":", alpha=0.45)

    x = np.arange(len(stress))
    axes[1, 1].bar(x, stress.Sparse_client_MSE,
                   color=[FED_RED, PALETTE[1], PALETTE[2]], width=0.68)
    ax3 = axes[1, 1].twinx()
    ax3.plot(x, stress.Round_traffic_MB, color="#4D4D4D",
             marker="o", ls="--")
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(stress.Scenario, rotation=18, ha="right")
    axes[1, 1].set_ylabel("Data-poor normalized MSE (-)")
    ax3.set_ylabel("Model traffic per round (MB)")
    axes[1, 1].grid(axis="y", ls=":", alpha=0.45)
    for ax, label in zip(axes.ravel(), "abcd"):
        panel_label(ax, f"({label})")
    fig.subplots_adjust(left=0.09, right=0.91, bottom=0.16, top=0.98,
                        wspace=0.42, hspace=0.47)
    fig.savefig(OUT / "r1_federation_scaling.pdf")
    plt.close(fig)


def plot_deployment_figure():
    sensitivity = pd.read_csv(OUT / "r1_obstacle_sensitivity_summary.csv")
    candidates = pd.read_csv(OUT / "r1_candidate_latency.csv")
    mc = pd.read_csv(OUT / "r1_mc_closed_loop_summary.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 4.75))
    for ax, parameter, xlabel in zip(
            axes.ravel()[:4], ["q_obs", "c_s", "inner_a", "d_inf"],
            [r"Obstacle weight $q_{obs}$", r"Outer coefficient $c_s$",
             r"Inner coefficient $a$", r"Influence distance $d_{inf}$ (m)"]):
        sub = sensitivity[sensitivity.Parameter == parameter]
        ax.plot(sub.Value, 100 * sub.SR, marker="o", color=FED_RED,
                label="SR (%)")
        ax.plot(sub.Value, sub.SC, marker="s", color=PALETTE[1],
                label="SC (%)")
        ax.set_xlabel(xlabel); ax.set_ylabel("Endpoint (%)")
        ax.grid(ls=":", alpha=0.45)
    axes[0, 0].legend(ncol=2, frameon=False)
    for M, sub in candidates.groupby("M"):
        axes[1, 1].plot(sub.Candidates, sub.Mean_ms, marker="o",
                        label=f"M={int(M)}")
    axes[1, 1].axhline(100, color="#4D4D4D", ls="--", lw=0.9)
    axes[1, 1].set_xlabel("Candidates per control call")
    axes[1, 1].set_ylabel("Candidate-evaluation latency (ms)")
    axes[1, 1].grid(ls=":", alpha=0.45)
    axes[1, 1].legend(ncol=2, frameon=False)
    axes[1, 2].scatter(mc.Mean_controller_ms, mc.SC_pct,
                       s=40 + 2 * mc.M, c=mc.M, cmap=POSITIVE,
                       edgecolor="#4D4D4D", linewidth=0.5)
    for _, row in mc.iterrows():
        axes[1, 2].annotate(f"M={int(row.M)}",
                            (row.Mean_controller_ms, row.SC_pct),
                            xytext=(3, 2), textcoords="offset points",
                            fontsize=6.5)
    axes[1, 2].set_xlabel("Mean controller latency (ms)")
    axes[1, 2].set_ylabel("Safety compliance (%)")
    axes[1, 2].grid(ls=":", alpha=0.45)
    for ax, label in zip(axes.ravel(), "abcdef"):
        panel_label(ax, f"({label})")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.13, top=0.98,
                        wspace=0.43, hspace=0.45)
    fig.savefig(OUT / "r1_deployment_sensitivity.pdf")
    plt.close(fig)


def write_metadata(prox):
    metadata = {
        "date": "2026-11-03",
        "test_seeds": TEST_SEEDS,
        "mc_test_seeds": MC_TEST_SEEDS,
        "sensitivity_seeds": SENSITIVITY_SEEDS,
        "federation_training_seeds": FEDERATION_SEEDS,
        "device": str(Config.DEVICE),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_parameters": sum(p.numel() for p in prox.parameters()),
        "model_payload_bytes": model_bytes(prox),
        "uncertainty": {
            "definition": "sqrt(trace(D^-1 Sigma D^-1))",
            "delta_scale": prox.delta_scale.detach().cpu().numpy().tolist(),
            "units": "dimensionless",
            "mc_covariance_ddof": 1,
        },
        "controller": {
            "horizon": Config.HORIZON,
            "control_period_s": Config.DT,
            "mc_samples_primary": Config.MC_SAMPLES,
            "safety_margin_m": Config.SAFETY_MARGIN,
            "uncertainty_weight": Config.UNCERTAINTY_WEIGHT,
            "uncertainty_margin_scale": Config.UNCERTAINTY_MARGIN_SCALE,
            "trust_uncertainty_scale": Config.BNN_TRUST_UNC_SCALE,
            "obstacle_penalty": Config.OBSTACLE_PENALTY,
            "obstacle_influence_m": Config.FED_OBSTACLE_INFLUENCE,
            "outer_obstacle_weight": Config.FED_SOFT_OBS_WEIGHT,
            "inner_obstacle_weight": Config.FED_INNER_OBS_WEIGHT,
        },
        "geometry": {
            "analytical_benchmark": "point center to circular-obstacle boundary",
            "collision": "signed clearance < 0 m",
            "compliance": f"signed clearance >= {Config.SAFETY_MARGIN} m",
            "MuJoCo": "simulator body-contact events and body-center clearance",
        },
        "statistics": {
            "independent_cluster": "environment seed",
            "routes_per_method": 4 * len(TEST_SEEDS),
            "standard_deviation": "sample SD (ddof=1)",
            "confidence_interval": "10,000-resample paired seed-cluster bootstrap",
            "binary_success_test": "exact McNemar/binomial test on paired routes",
        },
    }
    (OUT / "r1_metadata.json").write_text(json.dumps(metadata, indent=2),
                                          encoding="utf-8")


def main():
    Config.set_mode("fast")
    Config.SEEDS = TEST_SEEDS
    Config.N_SEEDS = len(TEST_SEEDS)
    data, metadata, prox, fedavg, central, local, ensemble = prepare_primary_models()
    write_metadata(prox)
    uncertainty, _ = uncertainty_diagnostics(prox, metadata)
    primary_and_mismatch(prox, central, local, ensemble, fedavg)
    mechanism_isolation(prox, uncertainty.Uncertainty.to_numpy())
    mc_sample_sweep(prox, metadata)
    candidate_scaling(prox)
    fleet_scaling()
    obstacle_sensitivity(prox)
    communication_budget(prox)
    plot_main_figure()
    plot_mechanism_figure()
    plot_federation_figure()
    plot_deployment_figure()
    print(f"[done] R1 artifacts -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
