"""Development-only study of physics-residual federated prediction.

The publication test seeds are deliberately absent.  Models and results from
this file stay under ``results_r1_dev`` until one configuration is frozen.
"""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import r1_experiments as r1
from config import Config
from controllers import FedUMPCController
from federated import FedAvgClient, ProxFLClient
from main import D_OBS, _gen_data, run_simulation
from r1_dev_selector import DEVELOPMENT_SEEDS, ScreenedFedUMPC
from utils import CrossingEnv, VehicleModel


ROOT = R1_ROOT
OUT = artifact_path("results_r1_dev")
MODEL_DIR = OUT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def residual_training_data():
    r1.reset_rng(100)
    original, metadata = _gen_data(return_metadata=True)
    nominal = VehicleModel(mass=Config.MASS, drag=Config.DRAG)
    transformed = {}
    for client, samples in original.items():
        transformed[client] = []
        for state, action, next_state in samples:
            nominal_next = nominal.step(state, action)
            residual = next_state - nominal_next
            transformed[client].append((state, action, state + residual))
    return transformed, metadata


def prepare_models():
    prox_path = MODEL_DIR / "residual_prox.pt"
    avg_path = MODEL_DIR / "residual_fedavg.pt"
    data, metadata = residual_training_data()
    residual_scale = (0.35, 0.35, 0.22, 0.12)
    with r1.temporary_config(BNN_DELTA_SCALE=residual_scale):
        if not prox_path.exists():
            print("[development] training residual Prox-FL", flush=True)
            prox, _, drift = r1.train_federated(ProxFLClient, data, seed=1666)
            r1.save_model(prox_path, prox)
            print(f"[development] residual Prox-FL drift={drift:.4f}", flush=True)
        if not avg_path.exists():
            print("[development] training residual FedAvg", flush=True)
            avg, _, drift = r1.train_federated(FedAvgClient, data, seed=1666)
            r1.save_model(avg_path, avg)
            print(f"[development] residual FedAvg drift={drift:.4f}", flush=True)
    prox = r1.load_model(prox_path)
    avg = r1.load_model(avg_path)
    return prox, avg, metadata


class ResidualRolloutMixin:
    """Interpret the federated model output as a correction to nominal physics."""

    def _safe_bnn_rollout_batch(self, u_batch, state):
        n_candidates = u_batch.shape[0]
        physics = np.empty((n_candidates, self.H, 4), dtype=float)
        for j in range(n_candidates):
            physics[j] = self._rollout(u_batch[j], state)

        hybrid = np.empty_like(physics)
        uncertainty = np.empty((n_candidates, self.H), dtype=np.float32)
        curr = np.repeat(np.asarray(state, dtype=float)[None, :],
                         n_candidates, axis=0)
        initial = curr.copy()
        for h in range(self.H):
            residual, variance = self.bnn.predict_horizon(curr, u_batch[:, h, :])
            residual = np.clip(residual, -0.8, 0.8)
            u_std = np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
            u_std = self._transform_uncertainty(u_std, h)
            uncertainty[:, h] = u_std
            previous_physics = initial if h == 0 else physics[:, h - 1, :]
            physics_increment = physics[:, h, :] - previous_physics
            trust = (np.exp(-u_std / max(Config.BNN_TRUST_UNC_SCALE, 1e-6))
                     if self.use_model_trust else np.ones_like(u_std))
            trust = np.clip(trust, 0.0, 0.75)[:, None]
            curr = curr + physics_increment + trust * residual
            curr[:, 0] = np.clip(curr[:, 0], -Config.WORLD_LIMIT,
                                 Config.WORLD_LIMIT)
            curr[:, 1] = np.clip(curr[:, 1], -Config.WORLD_LIMIT,
                                 Config.WORLD_LIMIT)
            hybrid[:, h, :] = curr
        return hybrid, uncertainty, physics


class ResidualFedUMPC(ResidualRolloutMixin, FedUMPCController):
    pass


class ResidualScreenedFedUMPC(ResidualRolloutMixin, ScreenedFedUMPC):
    pass


def evaluate():
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.USE_SLSQP_REFINEMENT = False
    prox, fedavg, metadata = prepare_models()
    configurations = {
        "residual_prox_original": (ResidualFedUMPC, prox, {}),
        "residual_avg_original": (ResidualFedUMPC, fedavg, {}),
        "residual_prox_screen6": (ResidualScreenedFedUMPC, prox,
                                  {"screen_horizon": 6,
                                   "contact_buffer": 0.05}),
        "residual_prox_screen10": (ResidualScreenedFedUMPC, prox,
                                   {"screen_horizon": 10,
                                    "contact_buffer": 0.05}),
        "residual_avg_screen10": (ResidualScreenedFedUMPC, fedavg,
                                  {"screen_horizon": 10,
                                   "contact_buffer": 0.05}),
    }
    path = OUT / "residual_development_runs.csv"
    if path.exists():
        rows = pd.read_csv(path).to_dict("records")
    else:
        rows = []
    completed = {(str(row["Configuration"]), int(row["Seed"]))
                 for row in rows}
    for name, (controller_class, model, kwargs) in configurations.items():
        for seed in DEVELOPMENT_SEEDS:
            if (name, int(seed)) in completed:
                continue
            print(f"[development] {name} seed={seed}", flush=True)
            r1.reset_rng(seed)
            env = CrossingEnv(seed=seed, obstacle_density=D_OBS)
            controllers = [controller_class(env, model, **kwargs)
                           for _ in range(4)]
            result = run_simulation(controllers, env, seed=seed, strict=True)
            rows.append({
                "Configuration": name,
                "Seed": seed,
                "SR_pct": 100 * result["SR"],
                "SC_pct": result["SC"],
                "CR_pct": 100 * result["CR"],
                "MinClearance_m": result["MinClearance"],
                "MeanViolationDepth_m": result["MeanViolationDepth"],
                "ComputeMs": result["ComputeMs"],
            })
            pd.DataFrame(rows).to_csv(path, index=False)
            completed.add((name, int(seed)))
    frame = pd.DataFrame(rows)
    summary = (frame.groupby("Configuration", sort=False)
               .agg({"SR_pct": ["mean", "std"],
                     "SC_pct": ["mean", "std"],
                     "CR_pct": ["mean", "std"],
                     "MinClearance_m": ["mean", "std"],
                     "MeanViolationDepth_m": ["mean", "std"],
                     "ComputeMs": ["mean", "std"]}))
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(OUT / "residual_development_summary.csv",
                                 index=False)
    (OUT / "residual_development_protocol.json").write_text(json.dumps({
        "purpose": "development-only residual-model selection",
        "development_seeds": DEVELOPMENT_SEEDS,
        "heldout_results_used_for_selection": False,
        "target": "next-state residual relative to nominal physics",
        "predictor_output_scale": list(prox.delta_scale.cpu().numpy()),
        "configurations": list(configurations),
        "torch_version": torch.__version__,
    }, indent=2), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    evaluate()
