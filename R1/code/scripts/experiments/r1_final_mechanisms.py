"""Run final removal and augmentation interventions for FedUMPC."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import r1_experiments as r1
from config import Config
from controllers import FedUMPCController


ROOT = R1_ROOT
OUT = artifact_path("results_r1_final_screened")
PROTOCOL = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
SEEDS = [int(value) for value in PROTOCOL["heldout_seeds"]]


class NominalClearanceFedUMPC(FedUMPCController):
    """Intervention that removes hybrid-rollout obstacle clearance only."""

    def _conservative_obstacle_cost(self, physics_state, hybrid_state, margin):
        d_phy = self.env.get_min_dist(physics_state[0], physics_state[1])
        return self._fed_obstacle_cost_from_clearance(d_phy, margin)

    def _screen_candidates(self, bnn_states, unc, phy_states):
        margins = self._margins(unc)
        clearances = np.empty((len(phy_states), self.H), dtype=float)
        for candidate in range(len(phy_states)):
            for horizon in range(self.H):
                state = phy_states[candidate, horizon]
                clearances[candidate, horizon] = self.env.get_min_dist(
                    state[0], state[1])
        hs = min(self.H, max(1, self.screen_horizon))
        worst_surplus = np.min(
            clearances[:, :hs] - margins[:, :hs], axis=1)
        worst_clearance = np.min(clearances[:, :hs], axis=1)
        admissible = np.flatnonzero(worst_surplus >= 0.0)
        noncontact = np.flatnonzero(worst_clearance >= self.contact_buffer)
        return admissible, noncontact, worst_surplus, worst_clearance


def verify() -> None:
    for name, expected in PROTOCOL["frozen_source_sha256"].items():
        if hashlib.sha256((source_file(name)).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Frozen source changed: {name}")


def main() -> None:
    verify()
    Config.SIM_STEPS = 200
    Config.HORIZON = 25
    Config.MC_SAMPLES = 10
    Config.USE_SLSQP_REFINEMENT = False
    Config.RESULTS_DIR = str(OUT)
    r1.OUT = OUT
    prox = r1.load_model(artifact_path("results_r1") / "models" / "prox.pt")
    methods = {
        "Full FedUMPC": lambda env: [
            FedUMPCController(env, prox) for _ in range(4)],
        "No uncertainty response": lambda env: [
            FedUMPCController(env, prox, uncertainty_aware=False)
            for _ in range(4)],
        "No bounded trust": lambda env: [
            FedUMPCController(env, prox, use_model_trust=False)
            for _ in range(4)],
        "No hybrid clearance": lambda env: [
            NominalClearanceFedUMPC(env, prox) for _ in range(4)],
        "+ nominal guard": lambda env: [
            FedUMPCController(env, prox, use_nominal_guard=True)
            for _ in range(4)],
        "No candidate screen": lambda env: [
            FedUMPCController(env, prox, use_candidate_screen=False)
            for _ in range(4)],
        "+ adaptive margin": lambda env: [
            FedUMPCController(env, prox, use_adaptive_margin=True)
            for _ in range(4)],
        "+ uncertainty cost": lambda env: [
            FedUMPCController(env, prox, use_risk_cost=True)
            for _ in range(4)],
    }
    runs, routes = r1.run_panel("final_mechanisms", methods, SEEDS)
    r1.summarize_runs(runs, "final_mechanisms")
    r1.paired_contrasts(
        runs, routes, reference="Full FedUMPC", prefix="final_mechanisms")
    sr = runs.SR * 100.0
    runs["CBS"] = (
        2.0 * sr * runs.SC / (sr + runs.SC).clip(lower=1e-12)
        * (1.0 - runs.CR)
    )
    table = (runs.groupby("Method", sort=False)
             .agg(SR=("SR", "mean"), SC=("SC", "mean"),
                  CR=("CR", "mean"), CBS=("CBS", "mean"))
             .reset_index())
    table.SR *= 100.0
    table.CR *= 100.0
    table.to_csv(OUT / "mechanism_table.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
