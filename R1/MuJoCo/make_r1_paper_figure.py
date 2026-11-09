"""Build the single-column 2x3 MuJoCo trajectory figure from saved states."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle
import numpy as np

import run_six_algorithms as core


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs_r1_screened" / "scene_median_development"
DESTINATION = (
    ROOT.parent.parent / "paper" / "IoTJ_FedUMPC" / "R1" / "figures"
    / "Experiment" / "R1" / "fig4_mujoco_trajectories.pdf"
)
AGENT_COLORS = ["#FF6666", "#FFAA53", "#50CC55", "#3399FF"]
DISPLAY_NAMES = {
    "Linear MPC": "Nominal MPC",
    "Stochastic MPC": "Scenario MPC",
    "Robust MPC": "Fixed-Margin MPC",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.linewidth": 0.55,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def checkerboard(ax) -> None:
    step = 4.0
    for ix, x in enumerate(np.arange(-22, 22, step)):
        for iy, y in enumerate(np.arange(-22, 22, step)):
            color = "#F2F2F2" if (ix + iy) % 2 == 0 else "#FFFFFF"
            ax.add_patch(Rectangle((x, y), step, step, facecolor=color,
                                   edgecolor="none", zorder=0))


def main() -> None:
    scenario = json.loads((OUTPUT / "scenario.json").read_text(encoding="utf-8"))
    records = {
        row["method"]: row
        for row in json.loads((OUTPUT / "metrics.json").read_text(encoding="utf-8"))
    }
    fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.68))
    starts = np.asarray(scenario["starts"], dtype=float)
    targets = np.asarray(scenario["targets"], dtype=float)
    for index, (ax, method) in enumerate(zip(axes.ravel(), core.METHODS)):
        checkerboard(ax)
        for obstacle in scenario["obstacles"]:
            ax.add_patch(Circle(
                (obstacle["x"], obstacle["y"]), obstacle["r"],
                facecolor="#C9CDD1", edgecolor="#5C6268", linewidth=0.45,
                zorder=2,
            ))
        safe = method.lower().replace(" ", "_").replace("-", "_")
        trajectory_payload = json.loads(
            (OUTPUT / f"{safe}_trajectories.json").read_text(encoding="utf-8")
        )
        for agent in range(4):
            trajectory = np.asarray(trajectory_payload[f"agent_{agent + 1}"], dtype=float)
            color = AGENT_COLORS[agent]
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color,
                    linewidth=0.85, zorder=3)
            ax.scatter(starts[agent, 0], starts[agent, 1], s=9, color=color,
                       edgecolor="white", linewidth=0.35, zorder=4)
            ax.scatter(targets[agent, 0], targets[agent, 1], s=10, marker="D",
                       facecolor="none", edgecolor=color, linewidth=0.65,
                       zorder=4)
            if not core.reached_goal(trajectory[-1], agent):
                ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=16,
                           marker="x", color="#202020", linewidth=0.8,
                           zorder=5)
        metric = records[method]
        ax.text(
            0.025, 0.025,
            f"S {int(metric['success'])}/4 | C {int(metric['collisions'])}/4\n"
            f"$d_{{min}}$ {metric['min_clearance']:.2f} m",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="#555555", linewidth=0.35, alpha=0.92),
            zorder=6,
        )
        display_name = DISPLAY_NAMES.get(method, method)
        ax.set_title(f"({chr(97 + index)}) {display_name}", fontsize=7.5,
                     fontweight="bold", loc="left", y=1.015, pad=1.0)
        ax.set_xlim(-22, 22); ax.set_ylim(-22, 22)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0, 0].plot([9, 19], [18, 18], color="#333333", lw=1.0)
    axes[0, 0].text(14, 14.5, "10 m", ha="center", va="top", fontsize=7.5)
    handles = [Line2D([], [], color=c, lw=1.0, label=f"Route {i+1}")
               for i, c in enumerate(AGENT_COLORS)]
    handles.append(Line2D([], [], color="#202020", marker="x", linestyle="none",
                          markeredgewidth=0.8, label="Terminated"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=7.5, handlelength=1.1, columnspacing=0.55,
               bbox_to_anchor=(0.5, -0.012))
    fig.subplots_adjust(left=0.02, right=0.995, bottom=0.10, top=0.945,
                        wspace=0.065, hspace=0.10)
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(DESTINATION, facecolor="white")
    plt.close(fig)
    print(f"Wrote {DESTINATION}")


if __name__ == "__main__":
    main()
