"""Run the six paper controllers on one MuJoCo intersection scene.

The controller receives state measured from MuJoCo, computes a receding-horizon
action, applies planar forces/torque to the MuJoCo vehicle, and records the
resulting MuJoCo body positions.  The final trail is inserted into MuJoCo's
render scene before the pixels are captured; it is not copied from the old
Matplotlib experiment.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
FRAMES = OUTPUT / "final_frames"

DT_CONTROL = 0.10
DT_PHYSICS = 0.01
SUBSTEPS = int(round(DT_CONTROL / DT_PHYSICS))
HORIZON = 18
MAX_STEPS = 190
WHEELBASE = 2.5
MAX_STEER = 0.60
MAX_ACCEL = 2.0
WORLD_LIMIT = 22.0
SAFETY_MARGIN = 1.35

METHODS = (
    "Linear MPC",
    "Tube MPC",
    "Stochastic MPC",
    "GP-MPC",
    "Robust MPC",
    "FedRMPC",
)

STARTS = np.array(
    [
        [-20.0, -2.0, 3.0, 0.0],
        [20.0, 2.0, 3.0, math.pi],
        [-2.0, -20.0, 3.0, math.pi / 2.0],
        [2.0, 20.0, 3.0, -math.pi / 2.0],
    ],
    dtype=float,
)
TARGETS = np.array(
    [
        [20.0, -2.0, 3.0, 0.0],
        [-20.0, 2.0, 3.0, math.pi],
        [-2.0, 20.0, 3.0, math.pi / 2.0],
        [2.0, -20.0, 3.0, -math.pi / 2.0],
    ],
    dtype=float,
)
AGENT_RGBA = np.array(
    [
        [0.92, 0.18, 0.16, 1.0],
        [0.10, 0.45, 0.88, 1.0],
        [0.12, 0.68, 0.30, 1.0],
        [0.95, 0.55, 0.08, 1.0],
    ],
    dtype=np.float32,
)


def wrap_angle(value: float | np.ndarray) -> float | np.ndarray:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def kinematic_step(state: np.ndarray, control: np.ndarray, dt: float = DT_CONTROL) -> np.ndarray:
    x, y, speed, yaw = state
    steer = float(np.clip(control[0], -MAX_STEER, MAX_STEER))
    accel = float(np.clip(control[1], -MAX_ACCEL, MAX_ACCEL))
    beta = math.atan(0.5 * math.tan(steer))
    next_state = np.array(
        [
            x + speed * math.cos(yaw + beta) * dt,
            y + speed * math.sin(yaw + beta) * dt,
            max(0.0, speed + (accel - 0.05 * speed) * dt),
            wrap_angle(yaw + speed / WHEELBASE * math.sin(beta) * dt),
        ],
        dtype=float,
    )
    return next_state


def generate_obstacles(seed: int) -> list[dict[str, float]]:
    obstacles = [
        {"x": -9.0, "y": -0.5, "r": 2.2},
        {"x": 9.0, "y": 0.5, "r": 2.2},
        {"x": -0.5, "y": -9.0, "r": 2.2},
        {"x": 0.5, "y": 9.0, "r": 2.2},
        {"x": 0.0, "y": 0.0, "r": 2.5},
    ]
    rng = np.random.default_rng(seed)
    while len(obstacles) < 11:
        x, y = rng.uniform(-16.0, 16.0, size=2)
        r = float(rng.uniform(1.1, 1.65))
        if np.linalg.norm([x, y]) < 10.0:
            continue
        if any(np.linalg.norm(np.array([x, y]) - s[:2]) < 4.0 for s in STARTS):
            continue
        if any(np.linalg.norm(np.array([x, y]) - t[:2]) < 4.0 for t in TARGETS):
            continue
        if any(np.linalg.norm(np.array([x, y]) - np.array([o["x"], o["y"]])) < r + o["r"] + 1.0
               for o in obstacles):
            continue
        obstacles.append({"x": float(x), "y": float(y), "r": r})
    return obstacles


def min_signed_distance(points: np.ndarray, obstacles: list[dict[str, float]]) -> np.ndarray:
    xy = np.atleast_2d(points)[..., :2]
    result = np.full(xy.shape[0], np.inf, dtype=float)
    for obs in obstacles:
        distance = np.linalg.norm(xy - np.array([obs["x"], obs["y"]]), axis=1) - obs["r"]
        result = np.minimum(result, distance)
    return result


def make_scene_xml(obstacles: list[dict[str, float]]) -> str:
    obstacle_xml = []
    for idx, obs in enumerate(obstacles):
        obstacle_xml.append(
            f'<geom name="obstacle_{idx}" type="cylinder" pos="{obs["x"]:.6f} {obs["y"]:.6f} 0.80" '
            f'size="{obs["r"]:.6f} 0.80" material="obstacle" contype="1" conaffinity="2"/>'
        )

    marker_xml = []
    for idx, (start, target) in enumerate(zip(STARTS, TARGETS)):
        color = AGENT_RGBA[idx]
        rgba = " ".join(f"{value:.3f}" for value in color)
        marker_xml.append(
            f'<site name="start_{idx}" type="cylinder" pos="{start[0]} {start[1]} 0.035" '
            f'size="0.70 0.025" rgba="{rgba}"/>'
        )
        marker_xml.append(
            f'<site name="goal_{idx}" type="cylinder" pos="{target[0]} {target[1]} 0.040" '
            f'size="0.90 0.030" rgba="{color[0]:.3f} {color[1]:.3f} {color[2]:.3f} 0.35"/>'
        )

    vehicle_xml = []
    actuator_xml = []
    for idx, start in enumerate(STARTS):
        color = AGENT_RGBA[idx]
        rgba = " ".join(f"{value:.3f}" for value in color)
        vehicle_xml.append(
            f"""
            <body name="vehicle_{idx}" pos="{start[0]} {start[1]} 0.38">
              <joint name="vehicle_{idx}_x" type="slide" axis="1 0 0" limited="true" range="-44 44"/>
              <joint name="vehicle_{idx}_y" type="slide" axis="0 1 0" limited="true" range="-44 44"/>
              <joint name="vehicle_{idx}_yaw" type="hinge" axis="0 0 1" damping="80"/>
              <geom name="vehicle_{idx}_body" type="box" size="1.10 0.55 0.28" mass="1000"
                    rgba="{rgba}" contype="2" conaffinity="1" friction="1.0 0.02 0.002"/>
              <geom type="box" pos="0.10 0 0.34" size="0.48 0.43 0.12" mass="1"
                    rgba="0.12 0.16 0.20 1" contype="0" conaffinity="0"/>
              <site name="vehicle_{idx}_nose" type="sphere" pos="0.92 0 0.30" size="0.12"
                    rgba="1 1 0.75 1"/>
            </body>
            """
        )
        actuator_xml.extend(
            [
                f'<motor name="vehicle_{idx}_fx" joint="vehicle_{idx}_x" gear="1" ctrllimited="true" ctrlrange="-12000 12000"/>',
                f'<motor name="vehicle_{idx}_fy" joint="vehicle_{idx}_y" gear="1" ctrllimited="true" ctrlrange="-12000 12000"/>',
                f'<motor name="vehicle_{idx}_tau" joint="vehicle_{idx}_yaw" gear="1" ctrllimited="true" ctrlrange="-6000 6000"/>',
            ]
        )

    return f"""
<mujoco model="FedRMPC intersection">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{DT_PHYSICS}" gravity="0 0 0" integrator="RK4" iterations="80"/>
  <size nconmax="300" njmax="1000"/>
  <visual>
    <global offwidth="1000" offheight="1000"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.01" zfar="150" fogstart="80" fogend="120"/>
    <headlight ambient="0.48 0.48 0.48" diffuse="0.72 0.72 0.72" specular="0.25 0.25 0.25"/>
  </visual>
  <asset>
    <texture name="ground_tex" type="2d" builtin="checker" rgb1="0.19 0.21 0.23" rgb2="0.24 0.26 0.28" width="512" height="512"/>
    <material name="ground" texture="ground_tex" texrepeat="12 12" reflectance="0.05"/>
    <material name="obstacle" rgba="0.63 0.42 0.22 1" specular="0.15" shininess="0.25"/>
  </asset>
  <worldbody>
    <light pos="0 -10 35" dir="0 0 -1" directional="true" diffuse="0.85 0.85 0.82"/>
    <geom name="ground" type="plane" size="24 24 0.1" material="ground" contype="0" conaffinity="0"/>
    <geom name="road_h" type="box" pos="0 0 0.015" size="22 3.4 0.015" rgba="0.10 0.11 0.12 1" contype="0" conaffinity="0"/>
    <geom name="road_v" type="box" pos="0 0 0.017" size="3.4 22 0.015" rgba="0.10 0.11 0.12 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="0 -0.05 0.045" size="22 0.035 0.012" rgba="0.95 0.82 0.20 0.75" contype="0" conaffinity="0"/>
    <geom type="box" pos="-0.05 0 0.047" size="0.035 22 0.012" rgba="0.95 0.82 0.20 0.75" contype="0" conaffinity="0"/>
    {''.join(obstacle_xml)}
    {''.join(marker_xml)}
    {''.join(vehicle_xml)}
  </worldbody>
  <actuator>
    {''.join(actuator_xml)}
  </actuator>
</mujoco>
"""


class NumpyDropoutBNN:
    """Small MC-dropout dynamics network trained by proximal federation."""

    def __init__(self, seed: int = 123, hidden: int = 36, dropout: float = 0.12):
        rng = np.random.default_rng(seed)
        self.dropout = dropout
        self.params = {
            "w1": rng.normal(0.0, 0.22, size=(7, hidden)),
            "b1": np.zeros(hidden),
            "w2": rng.normal(0.0, 0.18, size=(hidden, 4)),
            "b2": np.zeros(4),
        }
        self.input_mean = np.zeros(7)
        self.input_std = np.ones(7)
        self.output_mean = np.zeros(4)
        self.output_std = np.ones(4)
        self.rng = rng

    @staticmethod
    def features(states: np.ndarray, controls: np.ndarray) -> np.ndarray:
        states = np.atleast_2d(states)
        controls = np.atleast_2d(controls)
        return np.column_stack(
            [
                states[:, 0] / WORLD_LIMIT,
                states[:, 1] / WORLD_LIMIT,
                states[:, 2] / 10.0,
                np.sin(states[:, 3]),
                np.cos(states[:, 3]),
                controls[:, 0] / MAX_STEER,
                controls[:, 1] / MAX_ACCEL,
            ]
        )

    def _forward(self, x: np.ndarray, params: dict[str, np.ndarray], training: bool):
        z1 = x @ params["w1"] + params["b1"]
        hidden = np.maximum(z1, 0.0)
        mask = None
        if training:
            mask = (self.rng.random(hidden.shape) >= self.dropout).astype(float) / (1.0 - self.dropout)
            hidden = hidden * mask
        output = hidden @ params["w2"] + params["b2"]
        return output, (x, z1, hidden, mask)

    def _local_train(self, x: np.ndarray, y: np.ndarray, global_params: dict[str, np.ndarray],
                     seed: int) -> dict[str, np.ndarray]:
        self.rng = np.random.default_rng(seed)
        params = {name: value.copy() for name, value in global_params.items()}
        lr = 0.018
        proximal_mu = 0.025
        batch = 64
        for _ in range(7):
            order = self.rng.permutation(len(x))
            for start in range(0, len(x), batch):
                ids = order[start:start + batch]
                xb, yb = x[ids], y[ids]
                pred, cache = self._forward(xb, params, training=True)
                x0, z1, hidden, mask = cache
                grad_out = 2.0 * (pred - yb) / max(1, len(ids))
                grads = {
                    "w2": hidden.T @ grad_out,
                    "b2": grad_out.sum(axis=0),
                }
                grad_hidden = grad_out @ params["w2"].T
                if mask is not None:
                    grad_hidden *= mask
                grad_z1 = grad_hidden * (z1 > 0.0)
                grads["w1"] = x0.T @ grad_z1
                grads["b1"] = grad_z1.sum(axis=0)
                for name in params:
                    grads[name] += proximal_mu * (params[name] - global_params[name])
                    np.clip(grads[name], -4.0, 4.0, out=grads[name])
                    params[name] -= lr * grads[name]
        return params

    def fit_federated(self, seed: int = 123) -> None:
        rng = np.random.default_rng(seed)
        client_sets = []
        all_x, all_y = [], []
        for client in range(4):
            n = 360
            states = np.column_stack(
                [
                    rng.uniform(-21.0, 21.0, n),
                    rng.uniform(-21.0, 21.0, n),
                    rng.uniform(1.0, 8.0, n),
                    rng.uniform(-np.pi, np.pi, n),
                ]
            )
            controls = np.column_stack(
                [rng.uniform(-MAX_STEER, MAX_STEER, n), rng.uniform(-MAX_ACCEL, MAX_ACCEL, n)]
            )
            drag = 0.025 + 0.022 * client
            steer_gain = 0.91 + 0.06 * client
            increments = np.empty((n, 4), dtype=float)
            for row in range(n):
                s = states[row]
                u = controls[row].copy()
                u[0] *= steer_gain
                ns = kinematic_step(s, u)
                ns[2] += (0.05 - drag) * s[2] * DT_CONTROL
                ns[0] += 0.015 * np.sin(0.18 * s[1] + client)
                ns[1] += 0.015 * np.cos(0.18 * s[0] - client)
                increments[row] = ns - s
                increments[row, 3] = wrap_angle(increments[row, 3])
            x = self.features(states, controls)
            client_sets.append((x, increments))
            all_x.append(x)
            all_y.append(increments)

        full_x = np.vstack(all_x)
        full_y = np.vstack(all_y)
        self.input_mean = full_x.mean(axis=0)
        self.input_std = np.maximum(full_x.std(axis=0), 1e-5)
        self.output_mean = full_y.mean(axis=0)
        self.output_std = np.maximum(full_y.std(axis=0), 1e-5)
        client_sets = [
            ((x - self.input_mean) / self.input_std, (y - self.output_mean) / self.output_std)
            for x, y in client_sets
        ]

        global_params = {name: value.copy() for name, value in self.params.items()}
        for round_idx in range(9):
            locals_ = [
                self._local_train(x, y, global_params, seed + 100 * round_idx + client)
                for client, (x, y) in enumerate(client_sets)
            ]
            global_params = {
                name: np.mean([local_params[name] for local_params in locals_], axis=0)
                for name in global_params
            }
        self.params = global_params

    def predict(self, state: np.ndarray, control: np.ndarray, samples: int = 10) -> tuple[np.ndarray, float]:
        x = self.features(state, control)
        x = (x - self.input_mean) / self.input_std
        outputs = []
        for _ in range(samples):
            normalized, _ = self._forward(x, self.params, training=True)
            outputs.append(normalized[0] * self.output_std + self.output_mean)
        stack = np.asarray(outputs)
        mean = stack.mean(axis=0)
        variance = float(np.mean(np.var(stack, axis=0)))
        return mean, variance


@dataclass
class SimulationMetrics:
    method: str
    success: int
    collisions: int
    mean_clearance: float
    min_clearance: float
    mean_steps: float


class OnlineResidualGP:
    def __init__(self):
        self.features: list[np.ndarray] = []
        self.residuals: list[float] = []

    @staticmethod
    def feature(state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return np.array(
            [state[0] / 22.0, state[1] / 22.0, state[2] / 8.0,
             math.sin(state[3]), math.cos(state[3]), control[0], control[1] / 2.0]
        )

    def update(self, state: np.ndarray, control: np.ndarray, actual: np.ndarray, predicted: np.ndarray):
        self.features.append(self.feature(state, control))
        self.residuals.append(float(np.linalg.norm(actual - predicted)))
        if len(self.features) > 90:
            self.features.pop(0)
            self.residuals.pop(0)

    def sigma(self, state: np.ndarray, control: np.ndarray) -> float:
        if len(self.features) < 8:
            return 0.12
        query = self.feature(state, control)
        distances = np.linalg.norm(np.asarray(self.features) - query, axis=1)
        nearest = np.argsort(distances)[: min(10, len(distances))]
        weights = np.exp(-distances[nearest] ** 2 / 0.35)
        values = np.asarray(self.residuals)[nearest]
        mean = np.sum(weights * values) / max(np.sum(weights), 1e-9)
        novelty = float(np.min(distances))
        return float(np.clip(mean + 0.12 * novelty, 0.05, 0.85))


class SamplingMPC:
    def __init__(self, method: str, obstacles: list[dict[str, float]], bnn: NumpyDropoutBNN | None):
        self.method = method
        self.obstacles = obstacles
        self.bnn = bnn
        self.prev_u = np.zeros(2)
        self.gp = OnlineResidualGP() if method == "GP-MPC" else None

    def update_model(self, state: np.ndarray, control: np.ndarray, actual: np.ndarray):
        if self.gp is not None:
            predicted = kinematic_step(state, control)
            residual_state = actual - predicted
            residual_state[3] = wrap_angle(residual_state[3])
            self.gp.update(state, control, residual_state, np.zeros(4))

    def _threat_waypoints(self, state: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
        start = state[:2]
        goal = target[:2]
        segment = goal - start
        length = float(np.linalg.norm(segment))
        if length < 1e-6:
            return []
        direction = segment / length
        perpendicular = np.array([-direction[1], direction[0]])
        threats = []
        lookahead = min(length, max(9.0, state[2] * DT_CONTROL * HORIZON + 6.0))
        for obstacle in self.obstacles:
            center = np.array([obstacle["x"], obstacle["y"]])
            along = float(np.dot(center - start, direction))
            relative = center - start
            lateral = abs(float(direction[0] * relative[1] - direction[1] * relative[0]))
            if 0.0 < along < lookahead and lateral < obstacle["r"] + 2.6:
                threats.append((along, center, obstacle["r"]))
        if not threats:
            return []
        _, center, radius = min(threats, key=lambda item: item[0])
        waypoints = []
        for extra in (1.45, 2.15, 2.95):
            offset = radius + extra
            waypoints.append(center + perpendicular * offset + direction * 1.2)
            waypoints.append(center - perpendicular * offset + direction * 1.2)
        return waypoints

    def _pursuit_sequence(self, state: np.ndarray, target: np.ndarray,
                          waypoint: np.ndarray | None, bias: float) -> np.ndarray:
        current = state.copy()
        sequence = np.zeros((HORIZON, 2), dtype=float)
        for h in range(HORIZON):
            use_waypoint = waypoint is not None and h < int(0.48 * HORIZON) and np.linalg.norm(current[:2] - waypoint) > 1.2
            goal = waypoint if use_waypoint else target[:2]
            desired = math.atan2(goal[1] - current[1], goal[0] - current[0])
            desired += bias * max(0.0, 1.0 - h / max(1, HORIZON - 1))
            yaw_error = float(wrap_angle(desired - current[3]))
            sequence[h, 0] = np.clip(1.15 * yaw_error, -MAX_STEER, MAX_STEER)
            distance = float(np.linalg.norm(current[:2] - goal))
            target_speed = min(target[2] + 1.2, max(2.0, 0.42 * distance))
            sequence[h, 1] = np.clip(0.85 * (target_speed - current[2]), -MAX_ACCEL, MAX_ACCEL)
            current = kinematic_step(current, sequence[h])
        return sequence

    def _candidates(self, state: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
        candidates = [self._pursuit_sequence(state, target, None, bias) for bias in (-0.45, -0.20, 0.0, 0.20, 0.45)]
        for waypoint in self._threat_waypoints(state, target):
            candidates.extend(self._pursuit_sequence(state, target, waypoint, bias) for bias in (-0.12, 0.0, 0.12))
        warm = self._pursuit_sequence(state, target, None, 0.0)
        warm[0] = self.prev_u
        candidates.append(warm)
        for sign in (-1.0, 1.0):
            s_turn = self._pursuit_sequence(state, target, None, 0.0)
            split = HORIZON // 2
            s_turn[:split, 0] = sign * 0.46
            s_turn[split:, 0] = -sign * 0.24
            candidates.append(s_turn)
        return candidates

    def _rollout(self, state: np.ndarray, sequence: np.ndarray) -> np.ndarray:
        states = np.empty((HORIZON, 4), dtype=float)
        current = state.copy()
        for h in range(HORIZON):
            current = kinematic_step(current, sequence[h])
            states[h] = current
        return states

    def _obstacle_penalty(self, states: np.ndarray, margins: np.ndarray, soft_weight: float = 350.0) -> float:
        distances = min_signed_distance(states[:, :2], self.obstacles)
        influence = margins + 3.0
        hard = np.maximum(margins - distances, 0.0)
        soft = np.maximum(influence - distances, 0.0)
        return float(9000.0 * np.sum(hard ** 2) + soft_weight * np.sum(soft ** 2))

    def _base_cost(self, states: np.ndarray, sequence: np.ndarray, target: np.ndarray) -> float:
        distance = np.linalg.norm(states[:, :2] - target[:2], axis=1)
        terminal = float(distance[-1] ** 2)
        delta_u = np.diff(np.vstack([self.prev_u, sequence]), axis=0)
        wall = np.maximum(np.abs(states[:, :2]) - 21.0, 0.0)
        return float(8.0 * np.sum(distance ** 2) + 32.0 * terminal
                     + 0.12 * np.sum(sequence ** 2) + 0.45 * np.sum(delta_u ** 2)
                     + 9000.0 * np.sum(wall ** 2))

    def _cost(self, state: np.ndarray, target: np.ndarray, sequence: np.ndarray) -> float:
        nominal = self._rollout(state, sequence)
        base = self._base_cost(nominal, sequence, target)
        if self.method == "Linear MPC":
            return base + self._obstacle_penalty(nominal, np.full(HORIZON, SAFETY_MARGIN), soft_weight=75.0)
        if self.method == "Tube MPC":
            h = np.arange(HORIZON, dtype=float)
            radii = 0.18 * (1.0 - 0.80 ** (h + 1.0)) / (1.0 - 0.80)
            return base + self._obstacle_penalty(nominal, SAFETY_MARGIN + radii, soft_weight=220.0)
        if self.method == "Stochastic MPC":
            total = 0.0
            sigma_points = ((0.0, 0.0), (0.12, 0.0), (-0.12, 0.0), (0.0, 0.12), (0.0, -0.12))
            for dx, dy in sigma_points:
                perturbed = nominal.copy()
                scale = np.sqrt(np.arange(1, HORIZON + 1))
                perturbed[:, 0] += dx * scale
                perturbed[:, 1] += dy * scale
                total += self._base_cost(perturbed, sequence, target)
                total += self._obstacle_penalty(perturbed, np.full(HORIZON, SAFETY_MARGIN + 0.18), soft_weight=180.0)
            return total / len(sigma_points)
        if self.method == "GP-MPC":
            sigmas = np.array([self.gp.sigma(s, u) for s, u in zip(nominal, sequence)])
            return base + self._obstacle_penalty(nominal, SAFETY_MARGIN + 2.2 * sigmas, soft_weight=280.0)
        if self.method == "Robust MPC":
            return base + self._obstacle_penalty(nominal, np.full(HORIZON, SAFETY_MARGIN + 0.60), soft_weight=390.0)
        if self.method == "FedRMPC":
            learned = np.empty_like(nominal)
            uncertainties = np.empty(HORIZON)
            current = state.copy()
            for h in range(HORIZON):
                mean_increment, variance = self.bnn.predict(current, sequence[h], samples=8)
                physics_next = kinematic_step(current, sequence[h])
                physics_increment = physics_next - current
                physics_increment[3] = wrap_angle(physics_increment[3])
                mismatch = float(np.linalg.norm(mean_increment - physics_increment))
                trust = float(np.clip(np.exp(-4.0 * variance) * np.exp(-1.2 * mismatch), 0.0, 0.55))
                current = current + (1.0 - trust) * physics_increment + trust * mean_increment
                current[3] = wrap_angle(current[3])
                learned[h] = current
                uncertainties[h] = math.sqrt(max(variance, 0.0))
            mismatch = np.linalg.norm(learned[:, :2] - nominal[:, :2], axis=1)
            inflate = 1.0 + 0.12 * np.arange(HORIZON)
            margins = SAFETY_MARGIN + 0.56 + np.clip(1.15 * uncertainties * inflate, 0.0, 1.60)
            uncertainty_cost = float(np.sum(12.0 * np.exp(-0.08 * np.arange(HORIZON)) * uncertainties))
            return base + self._obstacle_penalty(nominal, margins, soft_weight=520.0) + 5.0 * np.sum(mismatch ** 2) + uncertainty_cost
        raise ValueError(self.method)

    def action(self, state: np.ndarray, target: np.ndarray) -> np.ndarray:
        candidates = self._candidates(state, target)
        costs = [self._cost(state, target, candidate) for candidate in candidates]
        best = candidates[int(np.argmin(costs))]
        self.prev_u = np.clip(best[0], [-MAX_STEER, -MAX_ACCEL], [MAX_STEER, MAX_ACCEL])
        return self.prev_u.copy()


class MuJoCoPlant:
    def __init__(self, xml: str):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.body_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"vehicle_{i}") for i in range(4)]
        self.qvel_addresses = []
        self.qpos_addresses = []
        self.ctrl_ids = []
        for i in range(4):
            joints = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"vehicle_{i}_{suffix}")
                      for suffix in ("x", "y", "yaw")]
            self.qvel_addresses.append([int(self.model.jnt_dofadr[joint]) for joint in joints])
            self.qpos_addresses.append([int(self.model.jnt_qposadr[joint]) for joint in joints])
            self.ctrl_ids.append([mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"vehicle_{i}_{suffix}")
                                  for suffix in ("fx", "fy", "tau")])
        for i, start in enumerate(STARTS):
            self.data.qpos[self.qpos_addresses[i][2]] = start[3]
            vx = start[2] * math.cos(start[3])
            vy = start[2] * math.sin(start[3])
            addresses = self.qvel_addresses[i]
            self.data.qvel[addresses[0]] = vx
            self.data.qvel[addresses[1]] = vy
        mujoco.mj_forward(self.model, self.data)

    def state(self, agent: int) -> np.ndarray:
        body_id = self.body_ids[agent]
        x, y = self.data.xpos[body_id, :2]
        rotation = self.data.xmat[body_id].reshape(3, 3)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
        addresses = self.qvel_addresses[agent]
        vx = self.data.qvel[addresses[0]]
        vy = self.data.qvel[addresses[1]]
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        speed = max(0.0, float(np.dot(np.array([vx, vy]), forward)))
        return np.array([x, y, speed, yaw], dtype=float)

    def _set_force(self, agent: int, control: np.ndarray, active: bool):
        ids = self.ctrl_ids[agent]
        if not active:
            self.data.ctrl[ids] = 0.0
            addresses = self.qvel_addresses[agent]
            self.data.qvel[addresses] *= 0.88
            return
        state = self.state(agent)
        yaw = state[3]
        addresses = self.qvel_addresses[agent]
        velocity = np.array([self.data.qvel[addresses[0]], self.data.qvel[addresses[1]]])
        forward = np.array([math.cos(yaw), math.sin(yaw)])
        lateral = np.array([-math.sin(yaw), math.cos(yaw)])
        forward_speed = float(np.dot(velocity, forward))
        lateral_speed = float(np.dot(velocity, lateral))
        accel = float(np.clip(control[1], -MAX_ACCEL, MAX_ACCEL))
        longitudinal_force = 1000.0 * (accel - 0.05 * forward_speed)
        lateral_force = -1000.0 * 5.0 * lateral_speed
        force = longitudinal_force * forward + lateral_force * lateral
        steer = float(np.clip(control[0], -MAX_STEER, MAX_STEER))
        beta = math.atan(0.5 * math.tan(steer))
        desired_rate = forward_speed / WHEELBASE * math.sin(beta)
        yaw_rate = float(self.data.qvel[addresses[2]])
        yaw_torque = 520.0 * 10.0 * (desired_rate - yaw_rate)
        self.data.ctrl[ids[0]] = np.clip(force[0], -12000.0, 12000.0)
        self.data.ctrl[ids[1]] = np.clip(force[1], -12000.0, 12000.0)
        self.data.ctrl[ids[2]] = np.clip(yaw_torque, -6000.0, 6000.0)

    def step(self, controls: list[np.ndarray], active: list[bool]):
        for _ in range(SUBSTEPS):
            for agent in range(4):
                self._set_force(agent, controls[agent], active[agent])
            mujoco.mj_step(self.model, self.data)


def reached_goal(state: np.ndarray, agent: int) -> bool:
    distance = float(np.linalg.norm(state[:2] - TARGETS[agent, :2]))
    route = TARGETS[agent, :2] - STARTS[agent, :2]
    progress = float(np.dot(state[:2] - STARTS[agent, :2], route) / np.dot(route, route))
    return distance <= 3.2 or (progress >= 0.91 and distance <= 5.0)


def is_collision(plant: MuJoCoPlant, agent: int) -> bool:
    vehicle_geom = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_GEOM, f"vehicle_{agent}_body")
    for index in range(plant.data.ncon):
        contact = plant.data.contact[index]
        if contact.geom1 == vehicle_geom or contact.geom2 == vehicle_geom:
            other = contact.geom2 if contact.geom1 == vehicle_geom else contact.geom1
            name = mujoco.mj_id2name(plant.model, mujoco.mjtObj.mjOBJ_GEOM, other)
            if name and name.startswith("obstacle_"):
                return True
    return False


def run_method(method: str, xml: str, obstacles: list[dict[str, float]], bnn: NumpyDropoutBNN):
    plant = MuJoCoPlant(xml)
    controllers = [SamplingMPC(method, obstacles, bnn) for _ in range(4)]
    trajectories = [[plant.state(agent).copy()] for agent in range(4)]
    active = [True] * 4
    success = [False] * 4
    collision = [False] * 4
    controls = [np.zeros(2) for _ in range(4)]
    clearances: list[float] = []
    steps = np.zeros(4, dtype=int)

    for _ in range(MAX_STEPS):
        previous_states = [plant.state(agent) for agent in range(4)]
        for agent in range(4):
            if active[agent]:
                controls[agent] = controllers[agent].action(previous_states[agent], TARGETS[agent])
        plant.step(controls, active)
        for agent in range(4):
            state = plant.state(agent)
            trajectories[agent].append(state.copy())
            if active[agent]:
                steps[agent] += 1
                controllers[agent].update_model(previous_states[agent], controls[agent], state)
                clearances.append(float(min_signed_distance(state[:2], obstacles)[0]))
                if is_collision(plant, agent):
                    collision[agent] = True
                    active[agent] = False
                elif reached_goal(state, agent):
                    success[agent] = True
                    active[agent] = False
        if not any(active):
            break

    metrics = SimulationMetrics(
        method=method,
        success=int(sum(success)),
        collisions=int(sum(collision)),
        mean_clearance=float(np.mean(clearances)) if clearances else 0.0,
        min_clearance=float(np.min(clearances)) if clearances else 0.0,
        mean_steps=float(np.mean(steps)),
    )
    return plant, [np.asarray(trajectory) for trajectory in trajectories], metrics


def add_trails(renderer: mujoco.Renderer, trajectories: list[np.ndarray]):
    for agent, trajectory in enumerate(trajectories):
        if len(trajectory) < 2:
            continue
        stride = max(1, len(trajectory) // 110)
        points = trajectory[::stride]
        if not np.allclose(points[-1], trajectory[-1]):
            points = np.vstack([points, trajectory[-1]])
        for start, end in zip(points[:-1], points[1:]):
            if renderer.scene.ngeom >= renderer.scene.maxgeom:
                return
            geom = renderer.scene.geoms[renderer.scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.zeros(3, dtype=float),
                np.zeros(3, dtype=float),
                np.eye(3).reshape(-1),
                AGENT_RGBA[agent].copy(),
            )
            p0 = np.array([start[0], start[1], 0.12], dtype=float)
            p1 = np.array([end[0], end[1], 0.12], dtype=float)
            mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.105, p0, p1)
            renderer.scene.ngeom += 1


def render_final(plant: MuJoCoPlant, trajectories: list[np.ndarray], destination: Path):
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.0, 0.0, 0.0])
    camera.distance = 56.0
    camera.azimuth = 90.0
    camera.elevation = -90.0
    option = mujoco.MjvOption()
    option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
    with mujoco.Renderer(plant.model, height=900, width=900, max_geom=3000) as renderer:
        renderer.update_scene(plant.data, camera=camera, scene_option=option)
        add_trails(renderer, trajectories)
        pixels = renderer.render()
    Image.fromarray(pixels).save(destination)


def save_trajectory_data(method: str, trajectories: list[np.ndarray]):
    payload = {f"agent_{idx + 1}": trajectory.tolist() for idx, trajectory in enumerate(trajectories)}
    safe = method.lower().replace(" ", "_").replace("-", "_")
    (OUTPUT / f"{safe}_trajectories.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def make_comparison(metrics: list[SimulationMetrics], destination_png: Path, destination_pdf: Path):
    fig, axes = plt.subplots(1, 6, figsize=(24, 4.55), constrained_layout=True)
    for axis, metric in zip(axes, metrics):
        safe = metric.method.lower().replace(" ", "_").replace("-", "_")
        image = np.asarray(Image.open(FRAMES / f"{safe}.png"))
        axis.imshow(image)
        axis.axis("off")
        axis.set_title(metric.method, fontsize=14, fontweight="bold", pad=8)
        axis.text(
            0.5,
            0.015,
            f"Success {metric.success}/4  |  Collision {metric.collisions}/4",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            color="white",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "black", "alpha": 0.72, "edgecolor": "none"},
        )
    fig.savefig(destination_png, dpi=300, facecolor="white")
    try:
        fig.savefig(destination_pdf, dpi=300, facecolor="white")
    except PermissionError:
        # PDF previewers on Windows often lock the existing file. Preserve the
        # rebuilt result under an explicit name instead of terminating after
        # the PNG has already been updated.
        destination_pdf = destination_pdf.with_name(destination_pdf.stem + "_updated.pdf")
        fig.savefig(destination_pdf, dpi=300, facecolor="white")
        print(f"[MuJoCo] Existing PDF is locked; wrote {destination_pdf.name} instead.")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123, help="Common obstacle and training seed")
    parser.add_argument("--method", choices=METHODS, help="Run only one algorithm")
    parser.add_argument("--skip-bnn-training", action="store_true", help="Diagnostic only; do not use for the final figure")
    parser.add_argument("--rebuild-figure", action="store_true", help="Rebuild the 1x6 figure from existing captures and metrics")
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FRAMES.mkdir(parents=True, exist_ok=True)
    if args.rebuild_figure:
        records = json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
        metrics = [SimulationMetrics(**record) for record in records]
        make_comparison(metrics, OUTPUT / "mujoco_six_trajectories_1x6.png", OUTPUT / "mujoco_six_trajectories_1x6.pdf")
        print(f"[MuJoCo] Rebuilt figure: {OUTPUT / 'mujoco_six_trajectories_1x6.png'}")
        return
    obstacles = generate_obstacles(args.seed)
    xml = make_scene_xml(obstacles)
    (ROOT / "scene_seed123.xml").write_text(xml, encoding="utf-8")
    (OUTPUT / "scenario.json").write_text(
        json.dumps({"seed": args.seed, "starts": STARTS.tolist(), "targets": TARGETS.tolist(), "obstacles": obstacles}, indent=2),
        encoding="utf-8",
    )

    bnn = NumpyDropoutBNN(seed=args.seed)
    if not args.skip_bnn_training:
        print("[MuJoCo] Training the lightweight proximal federated MC-dropout dynamics model ...", flush=True)
        bnn.fit_federated(seed=args.seed)

    selected: Iterable[str] = (args.method,) if args.method else METHODS
    metrics: list[SimulationMetrics] = []
    for method in selected:
        print(f"[MuJoCo] Running {method} ...", flush=True)
        plant, trajectories, result = run_method(method, xml, obstacles, bnn)
        safe = method.lower().replace(" ", "_").replace("-", "_")
        render_final(plant, trajectories, FRAMES / f"{safe}.png")
        save_trajectory_data(method, trajectories)
        metrics.append(result)
        print(
            f"  success={result.success}/4 collision={result.collisions}/4 "
            f"min_clearance={result.min_clearance:.3f}m mean_steps={result.mean_steps:.1f}",
            flush=True,
        )

    metrics_path = OUTPUT / "metrics.json"
    metrics_path.write_text(json.dumps([metric.__dict__ for metric in metrics], indent=2), encoding="utf-8")
    if not args.method:
        make_comparison(metrics, OUTPUT / "mujoco_six_trajectories_1x6.png", OUTPUT / "mujoco_six_trajectories_1x6.pdf")
        print(f"[MuJoCo] Final figure: {OUTPUT / 'mujoco_six_trajectories_1x6.png'}")


if __name__ == "__main__":
    main()
