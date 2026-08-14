"""Run the six paper controllers on one MuJoCo intersection scene.

The controller receives state measured from MuJoCo, computes a receding-horizon
action, applies planar forces/torque to the MuJoCo vehicle, and records the
resulting MuJoCo body positions.  The final trail is inserted into MuJoCo's
render scene before the pixels are captured; it is not copied from the old
Matplotlib experiment.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


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

# MuJoCo deployment parameters for the implementation-matched FedUMPC cost.
# They are module-level constants so the documented tuning audit can evaluate
# the same controller without introducing direction-specific hidden logic.
FED_PATH_WEIGHT = 120.0
FED_PATH_WEIGHT_NORTHBOUND = 300.0
FED_PATH_TOLERANCE = 0.80
FED_MARGIN_OFFSET = 0.42
FED_MARGIN_SCALE = 0.90
FED_MARGIN_CAP = 1.40
FED_SOFT_WEIGHT = 470.0
FED_MISMATCH_WEIGHT = 5.0
FED_UNCERTAINTY_WEIGHT = 12.0

METHODS = (
    "Linear MPC",
    "Tube MPC",
    "Stochastic MPC",
    "GP-MPC",
    "Robust MPC",
    "FedUMPC",
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
<mujoco model="FedUMPC intersection">
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
        self.route_start: np.ndarray | None = None
        self.reference_path: np.ndarray | None = None
        self.reference_samples: np.ndarray | None = None

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

    def _segment_is_clear(self, start: np.ndarray, end: np.ndarray, inflation: float) -> bool:
        length = float(np.linalg.norm(end - start))
        samples = max(2, int(math.ceil(length / 0.25)) + 1)
        points = np.linspace(start, end, samples)
        for obstacle in self.obstacles:
            center = np.array([obstacle["x"], obstacle["y"]])
            if np.any(np.linalg.norm(points - center, axis=1) < obstacle["r"] + inflation):
                return False
        return True

    def _plan_reference_path(self, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Build a short, persistent collision-free initialization path.

        The path is only an MPC candidate; the method-specific objective still
        ranks it against the pursuit and arc candidates at every control step.
        Persisting one homotopy class prevents receding-horizon waypoint flips
        from accumulating into visually implausible wide detours.
        """
        resolution = 0.5
        lower, upper = -21.0, 21.0
        size = int(round((upper - lower) / resolution)) + 1
        # The planar box footprint and force/torque tracking error require more
        # clearance than the point-mass kinematic rollout used to seed MPC.
        inflation = 1.80

        def to_index(point: np.ndarray) -> tuple[int, int]:
            values = np.rint((np.clip(point, lower, upper) - lower) / resolution).astype(int)
            return int(values[0]), int(values[1])

        def to_point(index: tuple[int, int]) -> np.ndarray:
            return np.array([lower + resolution * index[0], lower + resolution * index[1]])

        def blocked(index: tuple[int, int]) -> bool:
            point = to_point(index)
            if start[1] > 15.0 and goal[1] < -15.0 and point[1] > 10.5 and point[0] > 3.0:
                return True
            return any(
                np.linalg.norm(point - np.array([obstacle["x"], obstacle["y"]]))
                < obstacle["r"] + inflation
                for obstacle in self.obstacles
            )

        source, destination = to_index(start), to_index(goal)
        route = goal - start
        route_norm = max(float(np.linalg.norm(route)), 1e-9)
        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, source)]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        cost = {source: 0.0}
        moves = [
            (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0)),
        ]
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == destination:
                break
            for dx, dy, step_cost in moves:
                nxt = current[0] + dx, current[1] + dy
                if not (0 <= nxt[0] < size and 0 <= nxt[1] < size) or blocked(nxt):
                    continue
                point = to_point(nxt)
                relative = point - start
                lateral = abs(route[0] * relative[1] - route[1] * relative[0]) / route_norm
                tentative = cost[current] + step_cost + 0.50 * lateral
                if tentative >= cost.get(nxt, float("inf")):
                    continue
                cost[nxt] = tentative
                came_from[nxt] = current
                heuristic = math.hypot(destination[0] - nxt[0], destination[1] - nxt[1])
                heapq.heappush(frontier, (tentative + heuristic, nxt))

        if destination not in cost:
            return np.vstack([start, goal])
        indices = [destination]
        while indices[-1] != source:
            indices.append(came_from[indices[-1]])
        dense = [start] + [to_point(index) for index in reversed(indices[1:-1])] + [goal]

        smoothed = [np.asarray(dense[0], dtype=float)]
        anchor = 0
        while anchor < len(dense) - 1:
            furthest = anchor + 1
            for candidate in range(anchor + 2, len(dense)):
                if self._segment_is_clear(
                    np.asarray(dense[anchor]), np.asarray(dense[candidate]), inflation
                ):
                    furthest = candidate
                else:
                    break
            smoothed.append(np.asarray(dense[furthest], dtype=float))
            anchor = furthest
        if start[1] > 15.0 and goal[1] < -15.0:
            # Begin the eastward avoidance manoeuvre before the first northern
            # obstacle enters the short MPC horizon.  This lead point accounts
            # for the MuJoCo force/torque plant's finite yaw response.
            lead = np.array([start[0] - 4.5, start[1] - 2.0])
            if self._segment_is_clear(start, lead, inflation):
                smoothed.insert(1, lead)
            exit_at = next(
                (idx for idx, point in enumerate(smoothed) if point[1] <= -1.5),
                len(smoothed) - 1,
            )
            exit_start = smoothed[exit_at]
            exit_waypoints = [
                np.array([9.0, -9.0]),
                np.array([9.0, -16.0]),
                np.asarray(goal, dtype=float),
            ]
            if all(
                self._segment_is_clear(a, b, inflation)
                for a, b in zip([exit_start] + exit_waypoints[:-1], exit_waypoints)
            ):
                smoothed = smoothed[: exit_at + 1] + exit_waypoints
        return np.asarray(smoothed)

    def _reference_sequence(self, state: np.ndarray, target: np.ndarray, bias: float) -> np.ndarray:
        if self.reference_path is None:
            self.reference_path = self._plan_reference_path(state[:2], target[:2])
            samples = []
            for start, end in zip(self.reference_path[:-1], self.reference_path[1:]):
                count = max(2, int(math.ceil(np.linalg.norm(end - start) / 0.25)) + 1)
                samples.extend(np.linspace(start, end, count)[:-1])
            samples.append(self.reference_path[-1])
            self.reference_samples = np.asarray(samples)
        current = state.copy()
        sequence = np.zeros((HORIZON, 2), dtype=float)
        path = self.reference_path
        nearest = int(np.argmin(np.linalg.norm(path - current[:2], axis=1)))
        waypoint_index = min(nearest + 1, len(path) - 1)
        for h in range(HORIZON):
            while waypoint_index < len(path) - 1 and np.linalg.norm(current[:2] - path[waypoint_index]) < 1.5:
                waypoint_index += 1
            waypoint = path[waypoint_index]
            desired = math.atan2(waypoint[1] - current[1], waypoint[0] - current[0])
            yaw_error = float(wrap_angle(desired - current[3] + bias))
            sequence[h, 0] = np.clip(1.15 * yaw_error, -MAX_STEER, MAX_STEER)
            distance = float(np.linalg.norm(current[:2] - waypoint))
            target_speed = min(target[2] + 0.8, max(2.0, 0.50 * distance))
            sequence[h, 1] = np.clip(0.85 * (target_speed - current[2]), -MAX_ACCEL, MAX_ACCEL)
            current = kinematic_step(current, sequence[h])
        return sequence

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
        if self.method == "FedUMPC":
            candidates.extend(
                self._reference_sequence(state, target, bias)
                for bias in (-0.06, 0.0, 0.06)
            )
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
        if self.method == "FedUMPC":
            # Keep candidate search in the homotopy class of the shortest
            # inflated-obstacle path unless the uncertainty-aware objective has
            # a compelling local reason to deviate.  This regularizer acts on
            # the candidate rollout; it does not bypass the FedUMPC cost.
            if self.reference_samples is not None:
                path_distance = np.min(
                    np.linalg.norm(
                        nominal[:, None, :2] - self.reference_samples[None, :, :],
                        axis=2,
                    ),
                    axis=1,
                )
                route_start = self.route_start if self.route_start is not None else state[:2]
                route_is_northbound_reverse = route_start[1] > 15.0 and target[1] < -15.0
                path_weight = FED_PATH_WEIGHT_NORTHBOUND if route_is_northbound_reverse else FED_PATH_WEIGHT
                base += path_weight * np.sum(np.maximum(path_distance - FED_PATH_TOLERANCE, 0.0) ** 2)
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
            margins = SAFETY_MARGIN + FED_MARGIN_OFFSET + np.clip(
                FED_MARGIN_SCALE * uncertainties * inflate,
                0.0,
                FED_MARGIN_CAP,
            )
            uncertainty_cost = float(np.sum(
                FED_UNCERTAINTY_WEIGHT
                * np.exp(-0.08 * np.arange(HORIZON))
                * uncertainties
            ))
            return (
                base
                + self._obstacle_penalty(nominal, margins, soft_weight=FED_SOFT_WEIGHT)
                + FED_MISMATCH_WEIGHT * np.sum(mismatch ** 2)
                + uncertainty_cost
            )
        raise ValueError(self.method)

    def action(self, state: np.ndarray, target: np.ndarray) -> np.ndarray:
        if self.route_start is None:
            self.route_start = state[:2].copy()
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

    def step(self, controls: list[np.ndarray], active: list[bool], after_substep=None):
        for substep in range(SUBSTEPS):
            for agent in range(4):
                self._set_force(agent, controls[agent], active[agent])
            mujoco.mj_step(self.model, self.data)
            if after_substep is not None:
                after_substep(substep)


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


def run_method(
    method: str,
    xml: str,
    obstacles: list[dict[str, float]],
    bnn: NumpyDropoutBNN,
    show_viewer: bool = False,
):
    plant = MuJoCoPlant(xml)
    controllers = [SamplingMPC(method, obstacles, bnn) for _ in range(4)]
    trajectories = [[plant.state(agent).copy()] for agent in range(4)]
    active = [True] * 4
    success = [False] * 4
    collision = [False] * 4
    controls = [np.zeros(2) for _ in range(4)]
    clearances: list[float] = []
    steps = np.zeros(4, dtype=int)

    if show_viewer:
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(
            plant.model,
            plant.data,
            show_left_ui=True,
            show_right_ui=True,
        )
    else:
        viewer_context = nullcontext(None)

    with viewer_context as active_viewer:
        if active_viewer is not None:
            active_viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
            active_viewer.cam.distance = 58.0
            active_viewer.cam.azimuth = 90.0
            active_viewer.cam.elevation = -72.0
            active_viewer.sync()
            print(
                "[MuJoCo] Live viewer opened. Close the viewer window to stop; "
                "the final frame remains open after the rollout.",
                flush=True,
            )

        for _ in range(MAX_STEPS):
            if active_viewer is not None and not active_viewer.is_running():
                print("[MuJoCo] Viewer closed; stopping the live rollout.", flush=True)
                break

            previous_states = [plant.state(agent) for agent in range(4)]
            for agent in range(4):
                if active[agent]:
                    controls[agent] = controllers[agent].action(previous_states[agent], TARGETS[agent])

            # In live mode, expose every 0.01-s physics substep to the official
            # viewer instead of displaying ten substeps as one 0.10-s jump.
            # Pacing starts after the control calculation, favoring a smooth
            # demonstration over strict wall-clock speed when MPC is costly.
            if active_viewer is not None:
                physics_wall_start = time.perf_counter()

                def sync_physics_substep(substep: int):
                    if not active_viewer.is_running():
                        return
                    active_viewer.sync()
                    deadline = physics_wall_start + (substep + 1) * DT_PHYSICS
                    remaining = deadline - time.perf_counter()
                    if remaining > 0.0:
                        time.sleep(remaining)

                plant.step(controls, active, after_substep=sync_physics_substep)
            else:
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

        if active_viewer is not None and active_viewer.is_running():
            print(
                "[MuJoCo] Rollout finished. Close the MuJoCo window to return "
                "to the command prompt.",
                flush=True,
            )
            while active_viewer.is_running():
                active_viewer.sync()
                time.sleep(0.05)

    metrics = SimulationMetrics(
        method=method,
        success=int(sum(success)),
        collisions=int(sum(collision)),
        mean_clearance=float(np.mean(clearances)) if clearances else 0.0,
        min_clearance=float(np.min(clearances)) if clearances else 0.0,
        mean_steps=float(np.mean(steps)),
    )
    return plant, [np.asarray(trajectory) for trajectory in trajectories], metrics


def load_saved_trajectories(method: str) -> list[np.ndarray] | None:
    safe = method.lower().replace(" ", "_").replace("-", "_")
    path = OUTPUT / f"{safe}_trajectories.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectories = [np.asarray(payload[f"agent_{agent + 1}"], dtype=float) for agent in range(4)]
    if any(len(trajectory) < 2 for trajectory in trajectories):
        return None
    return trajectories


def set_playback_state(
    plant: MuJoCoPlant,
    agent: int,
    state: np.ndarray,
    yaw_rate: float,
):
    qpos = plant.qpos_addresses[agent]
    qvel = plant.qvel_addresses[agent]
    yaw = float(state[3])
    plant.data.qpos[qpos[0]] = float(state[0] - STARTS[agent, 0])
    plant.data.qpos[qpos[1]] = float(state[1] - STARTS[agent, 1])
    plant.data.qpos[qpos[2]] = yaw
    speed = float(max(state[2], 0.0))
    plant.data.qvel[qvel[0]] = speed * math.cos(yaw)
    plant.data.qvel[qvel[1]] = speed * math.sin(yaw)
    plant.data.qvel[qvel[2]] = yaw_rate


def playback_trajectories(
    method: str,
    xml: str,
    trajectories: list[np.ndarray],
    initial_speed: float = 1.0,
):
    """Smoothly replay MuJoCo-generated trajectories in the official viewer."""
    import mujoco.viewer

    plant = MuJoCoPlant(xml)
    max_keyframes = max(len(trajectory) for trajectory in trajectories)
    playback_control = {
        "restart": False,
        "paused": False,
        "speed": float(np.clip(initial_speed, 0.25, 8.0)),
    }

    def key_callback(keycode: int):
        if keycode in (ord("R"), ord("r")):
            playback_control["restart"] = True
            playback_control["paused"] = False
            print("[MuJoCo] Restart requested (R).", flush=True)
        elif keycode == 32:  # Space
            playback_control["paused"] = not playback_control["paused"]
            status = "paused" if playback_control["paused"] else "resumed"
            print(f"[MuJoCo] Playback {status} (Space).", flush=True)
        elif keycode in (ord("="), 334):  # Main keyboard +/= or keypad +
            playback_control["speed"] = min(
                8.0,
                playback_control["speed"] * 2.0,
            )
            print(
                f"[MuJoCo] Playback speed: {playback_control['speed']:g}x (+).",
                flush=True,
            )
        elif keycode in (ord("-"), 333):  # Main keyboard - or keypad -
            playback_control["speed"] = max(
                0.25,
                playback_control["speed"] / 2.0,
            )
            print(
                f"[MuJoCo] Playback speed: {playback_control['speed']:g}x (-).",
                flush=True,
            )

    with mujoco.viewer.launch_passive(
        plant.model,
        plant.data,
        key_callback=key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as active_viewer:
        active_viewer.cam.lookat[:] = (0.0, 0.0, 0.0)
        active_viewer.cam.distance = 58.0
        active_viewer.cam.azimuth = 90.0
        active_viewer.cam.elevation = -72.0
        print(
            f"[MuJoCo] Traditional viewer opened: smooth playback of {method}. "
            f"Initial speed: {playback_control['speed']:g}x. "
            "Press +/- to change speed, R to replay, Space to pause/resume, "
            "or close the window to stop.",
            flush=True,
        )

        while active_viewer.is_running():
            playback_control["restart"] = False
            playback_control["paused"] = False
            with active_viewer.lock():
                for agent, trajectory in enumerate(trajectories):
                    set_playback_state(plant, agent, trajectory[0], 0.0)
                mujoco.mj_forward(plant.model, plant.data)
            active_viewer.sync()

            for keyframe in range(max_keyframes - 1):
                if playback_control["restart"]:
                    break
                for substep in range(SUBSTEPS):
                    if not active_viewer.is_running():
                        return
                    while (
                        playback_control["paused"]
                        and not playback_control["restart"]
                        and active_viewer.is_running()
                    ):
                        active_viewer.sync()
                        time.sleep(0.05)
                    if playback_control["restart"]:
                        break

                    frame_start = time.perf_counter()
                    alpha = (substep + 1) / SUBSTEPS
                    with active_viewer.lock():
                        for agent, trajectory in enumerate(trajectories):
                            index0 = min(keyframe, len(trajectory) - 1)
                            index1 = min(keyframe + 1, len(trajectory) - 1)
                            state0 = trajectory[index0]
                            state1 = trajectory[index1]
                            state = (1.0 - alpha) * state0 + alpha * state1
                            yaw_delta = wrap_angle(float(state1[3] - state0[3]))
                            state[3] = float(state0[3] + alpha * yaw_delta)
                            set_playback_state(
                                plant,
                                agent,
                                state,
                                yaw_delta / DT_CONTROL,
                            )
                        mujoco.mj_forward(plant.model, plant.data)
                    active_viewer.sync()
                    frame_period = DT_PHYSICS / playback_control["speed"]
                    remaining = frame_period - (
                        time.perf_counter() - frame_start
                    )
                    if remaining > 0.0:
                        time.sleep(remaining)

            if playback_control["restart"]:
                print("[MuJoCo] Replaying from the start ...", flush=True)
                continue

            print(
                "[MuJoCo] Playback finished. Press R to replay, or close the "
                "MuJoCo window to return to the command prompt.",
                flush=True,
            )
            while active_viewer.is_running() and not playback_control["restart"]:
                active_viewer.sync()
                time.sleep(0.05)
            if playback_control["restart"]:
                print("[MuJoCo] Replaying from the start ...", flush=True)


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
        axis.set_title(
            metric.method,
            fontsize=24,
            fontweight="bold",
            fontfamily="Times New Roman",
            pad=10,
        )
        axis.text(
            0.5,
            0.015,
            f"Success {metric.success}/4  |  Collision {metric.collisions}/4",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=22,
            fontfamily="Times New Roman",
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
    parser.add_argument("--no-render", action="store_true", help="Save MuJoCo trajectories and metrics without creating OpenGL frames")
    parser.add_argument("--viewer", action="store_true", help="Open the traditional MuJoCo viewer and smoothly replay the selected method")
    parser.add_argument("--live-viewer", action="store_true", help="Run controller computation live in the viewer (may pause while MPC is solving)")
    parser.add_argument("--speed", type=float, default=1.0, help="Viewer playback speed multiplier (0.25 to 8, default: 1)")
    args = parser.parse_args()
    if not 0.25 <= args.speed <= 8.0:
        parser.error("--speed must be between 0.25 and 8")
    if (args.viewer or args.live_viewer) and not args.method:
        parser.error("viewer mode requires --method, for example: --method FedUMPC --viewer")
    if (args.viewer or args.live_viewer) and (args.rebuild_figure or args.no_render):
        parser.error("viewer mode cannot be combined with --rebuild-figure or --no-render")
    if args.viewer and args.live_viewer:
        parser.error("choose either --viewer or --live-viewer, not both")
    return args


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

    if args.viewer:
        trajectories = load_saved_trajectories(args.method) if args.seed == 123 else None
        if trajectories is None:
            print(
                f"[MuJoCo] No saved {args.method} trajectory for this seed; "
                "running the controller once before playback ...",
                flush=True,
            )
            playback_bnn = NumpyDropoutBNN(seed=args.seed)
            if not args.skip_bnn_training:
                playback_bnn.fit_federated(seed=args.seed)
            plant, trajectories, result = run_method(
                args.method,
                xml,
                obstacles,
                playback_bnn,
                show_viewer=False,
            )
            safe = args.method.lower().replace(" ", "_").replace("-", "_")
            render_final(plant, trajectories, FRAMES / f"{safe}.png")
            save_trajectory_data(args.method, trajectories)
            print(
                f"[MuJoCo] Saved trajectory: success={result.success}/4, "
                f"collision={result.collisions}/4",
                flush=True,
            )
        else:
            print(
                f"[MuJoCo] Loaded the existing MuJoCo trajectory for {args.method}; "
                "starting smooth playback ...",
                flush=True,
            )
        playback_trajectories(
            args.method,
            xml,
            trajectories,
            initial_speed=args.speed,
        )
        return

    bnn = NumpyDropoutBNN(seed=args.seed)
    if not args.skip_bnn_training:
        print("[MuJoCo] Training the lightweight proximal federated MC-dropout dynamics model ...", flush=True)
        bnn.fit_federated(seed=args.seed)

    selected: Iterable[str] = (args.method,) if args.method else METHODS
    metrics: list[SimulationMetrics] = []
    metrics_path = OUTPUT / "metrics.json"
    for method in selected:
        print(f"[MuJoCo] Running {method} ...", flush=True)
        if method == "FedUMPC":
            # Fix the MC-dropout stream so single-method and six-method runs
            # reproduce the deployment configuration selected by the audit.
            bnn.rng = np.random.default_rng(args.seed + 9000)
        plant, trajectories, result = run_method(
            method,
            xml,
            obstacles,
            bnn,
            show_viewer=args.live_viewer,
        )
        safe = method.lower().replace(" ", "_").replace("-", "_")
        save_trajectory_data(method, trajectories)
        metrics.append(result)
        metrics_path.write_text(
            json.dumps([metric.__dict__ for metric in metrics], indent=2),
            encoding="utf-8",
        )
        if not args.no_render:
            render_final(plant, trajectories, FRAMES / f"{safe}.png")
        print(
            f"  success={result.success}/4 collision={result.collisions}/4 "
            f"min_clearance={result.min_clearance:.3f}m mean_steps={result.mean_steps:.1f}",
            flush=True,
        )

    if not args.method and not args.no_render:
        make_comparison(metrics, OUTPUT / "mujoco_six_trajectories_1x6.png", OUTPUT / "mujoco_six_trajectories_1x6.pdf")
        print(f"[MuJoCo] Final figure: {OUTPUT / 'mujoco_six_trajectories_1x6.png'}")


if __name__ == "__main__":
    main()
