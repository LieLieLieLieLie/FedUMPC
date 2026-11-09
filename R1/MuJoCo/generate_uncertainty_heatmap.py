"""Regenerate the paper uncertainty heatmap on the exact MuJoCo seed-123 map."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

import run_six_algorithms as core


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / "IoTJ_FedUMPC" / "figures" / "Experiment" / "exp1_uncertainty_heatmap.pdf"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--grid", type=int, default=140)
    parser.add_argument("--mc-samples", type=int, default=16)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def batched_uncertainty(
    model: core.NumpyDropoutBNN,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    samples: int,
) -> np.ndarray:
    states = np.column_stack(
        [
            x_grid.ravel(),
            y_grid.ravel(),
            np.full(x_grid.size, 3.0),
            np.full(x_grid.size, -np.pi / 2.0),
        ]
    )
    controls = np.zeros((x_grid.size, 2))
    features = model.features(states, controls)
    features = (features - model.input_mean) / model.input_std
    predictions = []
    for _ in range(samples):
        normalized, _ = model._forward(features, model.params, training=True)
        predictions.append(normalized * model.output_std + model.output_mean)
    variance = np.var(np.asarray(predictions), axis=0, ddof=1)
    return np.sqrt(np.sum(variance, axis=1)).reshape(x_grid.shape)


def main() -> None:
    args = parse_args()
    obstacles = core.generate_obstacles(args.seed)
    model = core.NumpyDropoutBNN(seed=args.seed)
    model.fit_federated(seed=args.seed)

    coordinates = np.linspace(-20.0, 20.0, args.grid)
    x_grid, y_grid = np.meshgrid(coordinates, coordinates)
    epistemic = batched_uncertainty(model, x_grid, y_grid, args.mc_samples)

    obstacle_risk = np.zeros_like(epistemic)
    for obstacle in obstacles:
        signed_distance = (
            np.hypot(x_grid - obstacle["x"], y_grid - obstacle["y"])
            - obstacle["r"]
        )
        obstacle_risk = np.maximum(
            obstacle_risk,
            np.exp(-np.maximum(signed_distance, 0.0) / core.SAFETY_MARGIN),
        )
    low, high = np.percentile(epistemic, [5.0, 99.0])
    normalized = np.clip((epistemic - low) / max(high - low, 1e-12), 0.0, 1.0)
    composite = np.clip(0.35 * normalized + 0.65 * obstacle_risk, 0.0, 1.0)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    color_map = mcolors.LinearSegmentedColormap.from_list(
        "white_to_fedred", ["#FFFFFF", "#FFE1E1", "#FF9B9B", "#FF4F4F"]
    )
    figure, axis = plt.subplots(figsize=(10.4, 8.7))
    mesh = axis.pcolormesh(
        x_grid,
        y_grid,
        composite,
        cmap=color_map,
        shading="auto",
        vmin=0.0,
        vmax=1.0,
    )
    colorbar = figure.colorbar(mesh, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(r"Safety-weighted uncertainty $\mathcal{Z}$", fontsize=22)
    colorbar.ax.tick_params(labelsize=17)
    for obstacle in obstacles:
        axis.add_patch(
            plt.Circle(
                (obstacle["x"], obstacle["y"]),
                obstacle["r"],
                edgecolor="black",
                facecolor="#D2B48C",
                alpha=0.72,
                linewidth=1.7,
                zorder=10,
            )
        )
    axis.set_xlabel("X (m)", fontsize=24)
    axis.set_ylabel("Y (m)", fontsize=24)
    axis.tick_params(labelsize=18, width=1.7, length=6)
    axis.set_xlim(-20.0, 20.0)
    axis.set_ylim(-20.0, 20.0)
    axis.set_aspect("equal")
    figure.tight_layout()

    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    figure.savefig(ROOT / "outputs" / "exp1_uncertainty_heatmap_seed123.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    np.savez_compressed(
        ROOT / "outputs" / "exp1_uncertainty_heatmap_seed123.npz",
        x=x_grid,
        y=y_grid,
        epistemic=epistemic,
        obstacle_risk=obstacle_risk,
        composite=composite,
        obstacles=np.asarray(
            [[item["x"], item["y"], item["r"]] for item in obstacles], dtype=float
        ),
    )
    print(f"Wrote {args.output} with {len(obstacles)} MuJoCo obstacles.")


if __name__ == "__main__":
    main()
