"""Matched data-scarcity experiment for the IoTJ revision.

The script compares Local Only, FedAvg, and Prox-FL under the same
uncertainty-aware MPC, rollout seeds, training rounds, local epochs, and
candidate budget.  It evaluates two per-client training budgets (400 and 50)
on an independent, client-matched held-out set.
"""

from __future__ import annotations

import json
import inspect
import os
from pathlib import Path
import random as py_random
import sys
import types

# Deterministic fallbacks for Windows hosts whose legacy crypto/socket
# providers fail during NumPy/Torch import.  The experiment uses no network I/O.
try:
    os.urandom(1)
except OSError:
    _fallback_rng = py_random.Random(20260814)

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

import numpy as np
import pandas as pd
import torch

from config import Config
from controllers import FedRMPCController
from main import MASS_DIST, _gen_data
from paper_experiments import (ROLLOUT_SEEDS, aggregate_runs,
                                  compact_metrics, reset_rng, train_global,
                                  train_local_models)
from federated import FedAvgClient, ProxFLClient
from utils import VehicleModel


OUT = Path(__file__).resolve().parent / "results"


def generate_holdout(metadata: dict, samples_per_client: int = 400,
                     seed: int = 20260814) -> dict[int, list[tuple]]:
    rng = np.random.default_rng(seed)
    data = {i: [] for i in range(4)}
    for i in range(4):
        plant = VehicleModel(mass=metadata[i]["mass_kg"],
                             drag=metadata[i]["drag"])
        for _ in range(samples_per_client):
            state = np.array([
                rng.uniform(-22, 22), rng.uniform(-22, 22),
                rng.uniform(0, 12), rng.uniform(-np.pi, np.pi),
            ])
            action = rng.uniform(-1, 1, 2)
            noise = rng.choice([0.0, 0.05, 0.15, 0.3],
                               p=[0.4, 0.3, 0.2, 0.1])
            disturbance = rng.choice([0.0, 1.0], p=[0.8, 0.2])
            # VehicleModel uses NumPy's global RNG for process noise.
            np.random.seed(int(rng.integers(0, 2**31 - 1)))
            next_state = plant.step(state, action, noise_std=float(noise),
                                    disturbance=float(disturbance))
            data[i].append((state, action, next_state))
    return data


def mse_on_samples(model, samples: list[tuple]) -> float:
    states = np.asarray([sample[0] for sample in samples])
    actions = np.asarray([sample[1] for sample in samples])
    targets = np.asarray([sample[2] - sample[0] for sample in samples])
    inputs = torch.as_tensor(np.c_[states, actions], dtype=torch.float32,
                             device=Config.DEVICE)
    outputs = torch.as_tensor(targets, dtype=torch.float32,
                              device=Config.DEVICE)
    model.eval()
    with torch.no_grad():
        return float(torch.mean((model(inputs) - outputs) ** 2).item())


def heldout_mse(method: str, models, holdout: dict[int, list[tuple]]) -> float:
    if method == "Local Only":
        errors = [mse_on_samples(models[i], holdout[i]) for i in range(4)]
        return float(np.mean(errors))
    pooled = [sample for i in range(4) for sample in holdout[i]]
    return mse_on_samples(models, pooled)


def evaluate_budget(full_data: dict[int, list[tuple]], holdout: dict,
                    budget: int) -> list[dict]:
    train_data = {i: full_data[i][:budget] for i in range(4)}
    print(f"[data scarcity] training budget={budget}/client", flush=True)
    local_models = train_local_models(train_data, seed=666)
    fedavg_model, _ = train_global(FedAvgClient, train_data, seed=666)
    prox_model, _ = train_global(ProxFLClient, train_data, seed=666)
    methods = {
        "Local Only": local_models,
        "FedAvg": fedavg_model,
        "Prox-FL": prox_model,
    }
    rows = []
    for method, models in methods.items():
        print(f"[data scarcity] rollout {budget}/client | {method}",
              flush=True)
        reset_rng(20260811)
        if method == "Local Only":
            ctrl_fn = lambda env, ms=models: [
                FedRMPCController(env, ms[i]) for i in range(4)
            ]
        else:
            ctrl_fn = lambda env, model=models: [
                FedRMPCController(env, model) for _ in range(4)
            ]
        aggregate = aggregate_runs(ctrl_fn, seeds=ROLLOUT_SEEDS)
        row = compact_metrics(method, aggregate)
        row.update({
            "Samples_per_client": budget,
            "Heldout_MSE": heldout_mse(method, models, holdout),
            "Heldout_samples_per_client": len(holdout[0]),
            "Training_seed": 666,
            "Rollout_seeds": ",".join(map(str, ROLLOUT_SEEDS)),
        })
        rows.append(row)
    return rows


def main() -> None:
    Config.set_mode("fast")
    Config.SEEDS = ROLLOUT_SEEDS
    Config.N_SEEDS = len(ROLLOUT_SEEDS)
    reset_rng(100)
    full_data, metadata = _gen_data(return_metadata=True)
    holdout = generate_holdout(metadata)
    OUT.mkdir(parents=True, exist_ok=True)
    if "--unequal-only" not in sys.argv:
        rows = []
        for budget in (400, 50):
            rows.extend(evaluate_budget(full_data, holdout, budget))
        frame = pd.DataFrame(rows)
        frame.to_csv(OUT / "exp12_data_scarcity.csv", index=False)
        (OUT / "exp12_data_scarcity_protocol.json").write_text(json.dumps({
            "training_samples_per_client": [400, 50],
            "heldout_samples_per_client": 400,
            "training_seed": 666,
            "heldout_seed": 20260814,
            "rollout_seeds": ROLLOUT_SEEDS,
            "rounds": Config.ROUNDS,
            "local_epochs": Config.LOCAL_EPOCHS,
            "batch_size": Config.BATCH_SIZE,
            "horizon": Config.HORIZON,
            "mc_samples": Config.MC_SAMPLES,
            "controller": "identical uncertainty-aware MPC for all learners",
            "training_clients": metadata,
        }, indent=2), encoding="utf-8")
        print(frame[["Samples_per_client", "Method", "Heldout_MSE",
                     "SR_mean_pct", "SC_mean_pct", "Collision_mean"]])

    # Unequal-coverage audit: one data-poor vehicle (50 samples) collaborates
    # with three data-rich vehicles (400 samples).  This probes the precise IoV
    # motivation for cross-client sharing without altering controller settings.
    unequal_budgets = [50, 400, 400, 400]
    unequal_data = {i: full_data[i][:unequal_budgets[i]] for i in range(4)}
    print(f"[unequal coverage] budgets={unequal_budgets}", flush=True)
    local_models = train_local_models(unequal_data, seed=666)
    fedavg_model, _ = train_global(FedAvgClient, unequal_data, seed=666)
    prox_model, _ = train_global(ProxFLClient, unequal_data, seed=666)
    unequal_rows = []
    for method, models in {
        "Local Only": local_models,
        "FedAvg": fedavg_model,
        "Prox-FL": prox_model,
    }.items():
        client_mses = ([mse_on_samples(models[i], holdout[i]) for i in range(4)]
                       if method == "Local Only" else
                       [mse_on_samples(models, holdout[i]) for i in range(4)])
        reset_rng(20260811)
        if method == "Local Only":
            ctrl_fn = lambda env, ms=models: [
                FedRMPCController(env, ms[i]) for i in range(4)
            ]
        else:
            ctrl_fn = lambda env, model=models: [
                FedRMPCController(env, model) for _ in range(4)
            ]
        aggregate = aggregate_runs(ctrl_fn, seeds=ROLLOUT_SEEDS)
        row = compact_metrics(method, aggregate)
        row.update({
            "Training_samples_by_client": ",".join(map(str, unequal_budgets)),
            "Heldout_MSE_client1_data_poor": client_mses[0],
            "Heldout_MSE_client2": client_mses[1],
            "Heldout_MSE_client3": client_mses[2],
            "Heldout_MSE_client4": client_mses[3],
            "Heldout_MSE_mean": float(np.mean(client_mses)),
            "Training_seed": 666,
            "Rollout_seeds": ",".join(map(str, ROLLOUT_SEEDS)),
        })
        unequal_rows.append(row)
    unequal_frame = pd.DataFrame(unequal_rows)
    unequal_frame.to_csv(OUT / "exp12_unequal_coverage.csv", index=False)
    print(unequal_frame[["Method", "Heldout_MSE_client1_data_poor",
                         "Heldout_MSE_mean", "SR_mean_pct", "SC_mean_pct",
                         "Collision_mean"]])


if __name__ == "__main__":
    main()
