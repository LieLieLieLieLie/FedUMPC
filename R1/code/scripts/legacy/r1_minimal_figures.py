"""Generate the four nonredundant, single-column figures for the minimal R1.

This script only visualizes frozen records. It does not rerun a controller,
change an endpoint, or select a favorable test scene.
"""
from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import json
from pathlib import Path

import r1_experiments  # installs the deterministic Windows entropy fallback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

ROOT = R1_ROOT
RESULTS = artifact_path("results_r1")
PAPER = ROOT.parents[1] / "paper" / "IoTJ_FedUMPC" / "R1"
OUT = PAPER / "figures" / "Experiment" / "Minimal"
OUT.mkdir(parents=True, exist_ok=True)
MUJOCO = ROOT.parents[1] / "MuJoCo" / "R1" / "outputs"

FED = "#FF6666"
FONT = 8.25
PALETTE = ["#FFAA53", "#50CC55", "#3399FF", "#6666FF",
           "#9933FF", "#00DDDD", "#4D4D4D"]
POS = LinearSegmentedColormap.from_list("positive", ["#FFFFFF", "#FF4F4F"])
DIV = LinearSegmentedColormap.from_list(
    "signed", ["#007FFF", "#FFFFFF", "#FF4F4F"])

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman"],
    "font.size": FONT, "axes.labelsize": FONT, "axes.titlesize": FONT,
    "xtick.labelsize": FONT, "ytick.labelsize": FONT,
    "legend.fontsize": FONT, "axes.linewidth": 0.65,
    "lines.linewidth": 0.9, "xtick.major.width": 0.65,
    "ytick.major.width": 0.65, "xtick.major.size": 2.8,
    "ytick.major.size": 2.8, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def grid():
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 4.75))
    fig.subplots_adjust(left=.145, right=.985, bottom=.09, top=.94,
                        wspace=.47, hspace=.64)
    return fig, axs.ravel()


def title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", pad=5.0)
    ax.tick_params(pad=1.5)
    ax.grid(True, axis="y", linestyle=":", linewidth=.45, alpha=.35)
    ax.set_axisbelow(True)


def save(fig, name):
    path = OUT / name
    fig.savefig(path, format="pdf")
    plt.close(fig)
    assert path.exists() and path.stat().st_size > 1000


def uncertainty():
    df = pd.read_csv(RESULTS / "r1_uncertainty_samples.csv")
    uq = float(df.Uncertainty.quantile(.25))
    eq = float(df.Prediction_error.quantile(.75))
    df["Quartile"] = pd.qcut(df.Uncertainty, 4,
                             labels=["Q1", "Q2", "Q3", "Q4"])
    fig, (a, b, c, d) = grid()
    h = a.hist2d(df.Uncertainty, df.Prediction_error, bins=34, cmap=POS,
                 cmin=1)
    a.axvline(uq, color="#4D4D4D", ls="--", lw=.7)
    a.axhline(eq, color="#4D4D4D", ls="--", lw=.7)
    a.set_xlabel(r"Uncertainty $\mathcal{U}$ (–)")
    a.set_ylabel("Prediction error (–) ↓")
    title(a, "a", "Error density")

    vals = [df.loc[df.Quartile == q, "Prediction_error"].to_numpy()
            for q in ["Q1", "Q2", "Q3", "Q4"]]
    vp = b.violinplot(vals, showmedians=True, showextrema=False)
    for body in vp["bodies"]:
        body.set_facecolor(FED); body.set_edgecolor("#4D4D4D"); body.set_alpha(.72)
    vp["cmedians"].set_color("#4D4D4D"); vp["cmedians"].set_linewidth(.8)
    b.set_xticks(range(1, 5), ["Q1", "Q2", "Q3", "Q4"])
    b.set_xlabel("Uncertainty quartile")
    b.set_ylabel("Prediction error (–) ↓")
    title(b, "b", "Conditional spread")

    bins = pd.qcut(df.Uncertainty, 10, duplicates="drop")
    grouped = df.groupby(bins, observed=True)
    x = grouped.Uncertainty.mean().to_numpy()
    for q, color, label in [(.50, PALETTE[1], "Median"),
                            (.75, PALETTE[2], "75th pct."),
                            (.90, FED, "90th pct.")]:
        y = grouped.Prediction_error.quantile(q).to_numpy()
        c.plot(x, y, marker="o", ms=2.8, color=color, label=label)
    c.set_xlabel(r"Uncertainty $\mathcal{U}$ (–)")
    c.set_ylabel("Error quantile (–) ↓")
    c.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=.82,
             ncol=1, handlelength=1.2, borderpad=0.15, loc="upper left")
    title(c, "c", "Conditional quantiles")

    rows = []
    for scenario, g in df.groupby("Scenario"):
        rho = g[["Uncertainty", "Prediction_error"]].corr(method="spearman").iloc[0, 1]
        fl = 100 * ((g.Uncertainty <= uq) & (g.Prediction_error >= eq)).mean()
        rows.append((scenario, rho, fl, g.Prediction_error.mean()))
    for j, (scenario, rho, fl, meanerr) in enumerate(rows):
        d.scatter(rho, fl, s=18 + 52 * meanerr / max(x[3] for x in rows),
                  color=PALETTE[j], edgecolor="#4D4D4D", lw=.45)
        offsets = {"Wind": (2, -11), "Mass shift": (3, 2),
                   "High noise": (3, -1), "Nominal": (-32, -1)}
        label = scenario.replace("Model error", "Mass shift")
        d.annotate(label, (rho, fl), xytext=offsets.get(label, (2, 2)),
                   textcoords="offset points")
    d.set_xlabel(r"Spearman $\rho_s$ ↑")
    d.set_ylabel("False-low frequency (%) ↓")
    d.margins(x=.08, y=.10)
    title(d, "d", "Scenario reliability")
    save(fig, "minimal_uncertainty.pdf")


def federation():
    df = pd.read_csv(RESULTS / "r1_fleet_scaling_summary.csv")
    colors = {"Local Only": PALETTE[0], "FedAvg": PALETTE[1],
              "Prox-FL": FED, "Centralized": PALETTE[2]}
    fig, (a, b, c, d) = grid()
    fig.subplots_adjust(left=.20, right=.985, bottom=.09, top=.865,
                        wspace=.48, hspace=.62)
    for method in ["Local Only", "FedAvg", "Prox-FL", "Centralized"]:
        g = df[df.Method == method].sort_values("N")
        a.errorbar(g.N, g.Sparse_client_MSE, yerr=g.Sparse_client_MSE_sd,
                   marker="o", ms=3, capsize=2, color=colors[method], label=method)
    a.set_xticks([4, 16, 32]); a.set_xlabel("Fleet size, N (vehicles)")
    a.set_ylabel("Data-poor MSE (–) ↓")
    title(a, "a", "Data-poor transfer")
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(.54, .995), columnspacing=.55, handlelength=1.1)

    for method in ["FedAvg", "Prox-FL"]:
        g = df[df.Method == method]
        b.scatter(g.Final_drift, g.Mean_MSE, s=24, color=colors[method],
                  edgecolor="#4D4D4D", lw=.45)
        label_offsets = {
            ("FedAvg", 4): (5, 10), ("FedAvg", 16): (5, -14),
            ("FedAvg", 32): (-22, 8),
            ("Prox-FL", 4): (5, 8), ("Prox-FL", 16): (12, -12),
            ("Prox-FL", 32): (-22, 0),
        }
        for _, row in g.iterrows():
            b.annotate(str(int(row.N)), (row.Final_drift, row.Mean_MSE),
                       xytext=label_offsets[(method, int(row.N))],
                       textcoords="offset points",
                       arrowprops=dict(arrowstyle="-", color="#666666",
                                       lw=.45, shrinkA=0, shrinkB=2))
    b.set_xlabel("Client drift (–) ↓")
    b.set_ylabel("Fleet MSE (–) ↓")
    b.set_ylim(.0082, .0302)
    title(b, "b", "Drift is not accuracy")

    piv = df[df.Method.isin(["Centralized", "FedAvg", "Prox-FL"])].pivot(
        index="Method", columns="N", values="Empirical_B").reindex(
            ["Centralized", "FedAvg", "Prox-FL"])
    im = c.imshow(piv.to_numpy(), cmap=POS, aspect="auto", vmin=0,
                  vmax=np.nanmax(piv.to_numpy()))
    c.set_xticks(range(3), ["4", "16", "32"])
    c.set_yticks(range(3), piv.index, rotation=35, ha="right")
    c.set_xlabel("Fleet size, N (vehicles)")
    for iy in range(3):
        for ix in range(3):
            c.text(ix, iy, f"{piv.iloc[iy, ix]:.2f}", ha="center", va="center")
    title(c, "c", r"Empirical $\widehat{B}$")
    c.grid(False)

    comm = pd.read_csv(RESULTS / "r1_communication_stress_summary.csv")
    order = ["Synchronous full", "50% participation", "50% + two-round stale"]
    markers = ["o", "s", "D"]
    for j, scenario in enumerate(order):
        row = comm[comm.Scenario == scenario].iloc[0]
        d.errorbar(row.Round_traffic_MB, row.Mean_MSE, yerr=row.Mean_MSE_sd,
                   marker=markers[j], ms=4, capsize=2, color=[FED, PALETTE[1], PALETTE[3]][j])
        d.annotate(["Full", "Half", "Half + stale"][j],
                   (row.Round_traffic_MB, row.Mean_MSE), xytext=(2, 2),
                   textcoords="offset points")
    d.set_xlabel("Payload per round (MB) ↓")
    d.set_ylabel("Fleet MSE (–) ↓")
    title(d, "d", "Communication stress")
    save(fig, "minimal_federation.pdf")


def deployment():
    mc = pd.read_csv(RESULTS / "r1_mc_query_diagnostics.csv")
    sens = pd.read_csv(RESULTS / "r1_obstacle_sensitivity_summary.csv")
    lat = pd.read_csv(RESULTS / "r1_candidate_latency.csv")
    fig, (a, b, c, d) = grid()
    a.plot(mc.M, mc.Mean_query_repeat_variance, color=FED, marker="o", ms=3.2)
    a.set_yscale("log"); a.set_xticks([5, 10, 20, 40])
    a.set_xlabel("MC samples, M")
    a.set_ylabel(r"Repeat variance of $\mathcal{U}$ (–) ↓")
    title(a, "a", "Sampling repeatability")

    labels, matrix = [], []
    for parameter, g in sens.groupby("Parameter", sort=False):
        g = g.sort_values("Value")
        base_row = g.iloc[int(np.argmin(np.abs(g.Value.to_numpy() - g.Base.to_numpy())))]
        base = base_row.SC
        labels.append({"c_s": r"$c_s$", "d_inf": r"$d_{inf}$",
                       "inner_a": r"$c_{in}$", "q_obs": r"$q_{obs}$"}[parameter])
        matrix.append((g.SC - base).to_numpy())
    arr = np.vstack(matrix)
    lim = max(abs(arr.min()), abs(arr.max()))
    b.imshow(arr, cmap=DIV, aspect="auto", vmin=-lim, vmax=lim)
    b.set_xticks([0, 1, 2], ["Low", "Base", "High"])
    b.set_yticks(range(4), labels)
    for iy in range(4):
        for ix in range(3):
            b.text(ix, iy, f"{arr[iy, ix]:+.1f}", ha="center", va="center")
    b.set_xlabel("One-factor setting")
    title(b, "b", "SC change (pp)")
    b.grid(False)

    for j, m in enumerate([5, 10, 20, 40]):
        g = lat[lat.M == m].sort_values("Candidates")
        c.plot(g.Candidates, g.Mean_ms, marker="o", ms=3, color=[PALETTE[0], PALETTE[1], PALETTE[2], FED][j], label=f"M={m}")
    c.axhline(100, color="#4D4D4D", ls="--", lw=.7)
    c.set_xticks([4, 8, 12, 16])
    c.set_xlabel("Candidates per call")
    c.set_ylabel("Mean latency (ms) ↓")
    c.legend(frameon=False, ncol=2, handlelength=1.1, columnspacing=.6)
    title(c, "c", "Candidate scaling")

    piv = lat.pivot(index="M", columns="Candidates", values="P95_ms").reindex([5, 10, 20, 40])
    d.imshow(piv.to_numpy(), cmap=POS, aspect="auto", vmin=0,
             vmax=piv.to_numpy().max())
    d.set_xticks(range(4), ["4", "8", "12", "16"])
    d.set_yticks(range(4), ["5", "10", "20", "40"])
    d.set_xlabel("Candidates per call")
    d.set_ylabel("MC samples, M")
    for iy in range(4):
        for ix in range(4):
            d.text(ix, iy, f"{piv.iloc[iy, ix]:.0f}", ha="center", va="center")
    title(d, "d", "P95 latency (ms) ↓")
    d.grid(False)
    save(fig, "minimal_deployment.pdf")


def mujoco():
    scenario = json.loads((MUJOCO / "scenario.json").read_text(encoding="utf-8"))
    fed = json.loads((MUJOCO / "fedumpc_trajectories.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(PAPER / "data" / "Experiment" / "Revision" / "r1_mujoco_multiseed.csv")
    names = ["Linear MPC", "Tube MPC", "Stochastic MPC", "GP-MPC", "Robust MPC", "FedUMPC"]
    files = {n: MUJOCO / ({"Linear MPC":"linear_mpc", "Tube MPC":"tube_mpc",
                           "Stochastic MPC":"stochastic_mpc", "GP-MPC":"gp_mpc",
                           "Robust MPC":"robust_mpc", "FedUMPC":"fedumpc"}[n] + "_trajectories.json")
             for n in names}
    method_colors = dict(zip(names, PALETTE[:5] + [FED]))
    fig, (a, b, c, d) = grid()
    fig.set_size_inches(3.5, 4.95)
    fig.subplots_adjust(left=.145, right=.985, bottom=.175, top=.94,
                        wspace=.47, hspace=.64)
    route_colors = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[4]]
    for obs in scenario["obstacles"]:
        a.add_patch(Circle((obs["x"], obs["y"]), obs["r"], facecolor="#D9D9D9",
                           edgecolor="#4D4D4D", lw=.45))
    for j, key in enumerate(sorted(fed)):
        tr = np.asarray(fed[key])
        a.plot(tr[:, 0], tr[:, 1], color=route_colors[j], lw=1.0)
        a.scatter(tr[0, 0], tr[0, 1], s=12, color=route_colors[j], marker="o")
        a.scatter(tr[-1, 0], tr[-1, 1], s=12, color=route_colors[j], marker="x")
    a.set_aspect("equal"); a.set_xlim(-22, 22); a.set_ylim(-22, 22)
    a.set_xlabel("x position (m)"); a.set_ylabel("y position (m)")
    title(a, "a", "Executed FedUMPC routes")

    route_key = "agent_2"
    for name in names:
        trj = json.loads(files[name].read_text(encoding="utf-8"))
        tr = np.asarray(trj[route_key])
        mask = (np.abs(tr[:, 0]) <= 8) & (np.abs(tr[:, 1]) <= 8)
        b.plot(tr[mask, 0], tr[mask, 1], color=method_colors[name], label=name)
    for obs in scenario["obstacles"][:5]:
        b.add_patch(Circle((obs["x"], obs["y"]), obs["r"], facecolor="#EEEEEE",
                           edgecolor="#4D4D4D", lw=.4))
    b.set_aspect("equal"); b.set_xlim(-8, 8); b.set_ylim(-8, 8)
    b.set_xlabel("x position (m)"); b.set_ylabel("y position (m)")
    title(b, "b", "Predesignated route-2 detail")

    for name in names:
        vals = np.sort(raw.loc[raw.method == name, "min_clearance"].to_numpy())
        c.step(vals, np.arange(1, len(vals) + 1) / len(vals), where="post",
               color=method_colors[name], label=name)
    c.axvline(1.35, color="#4D4D4D", ls="--", lw=.7)
    c.set_xlabel("Scene minimum clearance (m) ↑")
    c.set_ylabel("ECDF (–)")
    title(c, "c", "Five-scene clearance")

    for name in names:
        g = raw[raw.method == name]
        d.scatter(g.max_violation_duration_s, g.mean_violation_depth,
                  color=method_colors[name], s=15, alpha=.78, label=name,
                  edgecolor="#4D4D4D", lw=.3)
    d.set_xlabel("Longest violation (s) ↓")
    d.set_ylabel("Mean depth (m) ↓")
    title(d, "d", "Violation severity")
    handles, labels = b.get_legend_handles_labels()
    short = ["Nominal", "Tube", "Scenario", "GP-margin", "Fixed", "FedUMPC"]
    fig.legend(handles, short, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(.53, .01), columnspacing=.65, handlelength=1.1)
    save(fig, "minimal_mujoco.pdf")


if __name__ == "__main__":
    uncertainty()
    federation()
    deployment()
    mujoco()
    print(f"Wrote four PDF figures to {OUT}")
