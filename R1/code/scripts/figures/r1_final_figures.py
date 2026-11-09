"""Generate the R0-style, nonredundant R1 paper figures as vector PDFs."""

from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


ROOT = R1_ROOT
FINAL = artifact_path("results_r1_final_screened")
AUX = artifact_path("results_r1")
PAPER = ROOT.parents[1] / "paper" / "IoTJ_FedUMPC" / "R1"
OUT = PAPER / "figures" / "Experiment" / "R1"

FED = "#FF6666"
PALETTE = ["#FFAA53", "#50CC55", "#3399FF", "#6666FF",
           "#9933FF", "#00DDDD", "#4D4D4D"]
COLORS = {
    "Nominal MPC": PALETTE[0],
    "Tube-style MPC": PALETTE[1],
    "Scenario MPC": PALETTE[2],
    "GP-margin MPC": PALETTE[3],
    "Fixed-margin MPC": PALETTE[4],
    "Local UMPC": PALETTE[5],
    "FedAvg UMPC": PALETTE[6],
    "Ensemble UMPC": "#777777",
    "Centralized UMPC": "#222222",
    "FedUMPC": FED,
    "Local Only": PALETTE[0],
    "FedAvg": PALETTE[2],
    "Prox-FL": FED,
    "Centralized": PALETTE[6],
}
ALIASES = {
    "Nominal MPC": "Nom.",
    "Tube-style MPC": "Tube",
    "Scenario MPC": "Scen.",
    "GP-margin MPC": "GP",
    "Fixed-margin MPC": "Fixed",
    "Local UMPC": "Local",
    "FedAvg UMPC": "FedAvg",
    "Ensemble UMPC": "Ensemble",
    "Centralized UMPC": "Centralized",
    "FedUMPC": "FedUMPC",
}
METHODS = list(ALIASES)
SIX = ["Nominal MPC", "Tube-style MPC", "Scenario MPC",
       "GP-margin MPC", "Fixed-margin MPC", "FedUMPC"]
POS_CMAP = LinearSegmentedColormap.from_list(
    "white_red", ["#FFFFFF", "#FF4F4F"])
DIV_CMAP = LinearSegmentedColormap.from_list(
    "blue_white_red", ["#007FFF", "#FFFFFF", "#FF4F4F"])


def setup_style():
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.15,
        "lines.markersize": 3.5,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.bbox": None,
    })


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", loc="left", y=1.015, pad=1.5,
                 fontweight="bold")
    ax.grid(True, color="#E6E6E6", linewidth=0.45, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save(fig, name, *, top=0.94, bottom=0.11, hspace=0.45, wspace=0.38):
    OUT.mkdir(parents=True, exist_ok=True)
    right = 0.86 if name in {'fig7_federation.pdf', 'fig9_deployment_sensitivity.pdf', 'fig10_uncertainty_error.pdf'} else 0.98
    fig.subplots_adjust(left=0.15, right=right, top=top, bottom=bottom,
                        hspace=hspace, wspace=wspace)
    fig.savefig(OUT / name, format="pdf", bbox_inches=None)
    plt.close(fig)


def cbs(frame):
    sr = frame["SR"].to_numpy(dtype=float) * 100.0
    sc = frame["SC"].to_numpy(dtype=float)
    cr = frame["CR"].to_numpy(dtype=float)
    return 2 * sr * sc / np.maximum(sr + sc, 1e-12) * (1 - cr)


def percentile_ci(values, seed=20261104, n_boot=10000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), (n_boot, len(values)))].mean(1)
    return float(values.mean()), *np.percentile(means, [2.5, 97.5])


def common_plant_figure():
    runs = pd.read_csv(FINAL / "guarded_common_runs.csv")
    routes = pd.read_csv(FINAL / "guarded_common_routes.csv")
    runs["CBS"] = cbs(runs)
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 4.7))

    ax = axs[0, 0]
    reference = runs[runs.Method == "FedUMPC"].set_index("Seed")["CBS"]
    rows = []
    for j, method in enumerate(m for m in SIX if m != "FedUMPC"):
        comp = runs[runs.Method == method].set_index("Seed")["CBS"]
        delta = reference.loc[comp.index] - comp
        mean, lo, hi = percentile_ci(delta, seed=20261104 + j)
        rows.append((method, mean, lo, hi))
    rows.sort(key=lambda item: item[1])
    y = np.arange(len(rows))
    for yi, (method, mean, lo, hi) in zip(y, rows):
        if mean > 25.0:
            ax.annotate("", xy=(24.7, yi), xytext=(21.5, yi),
                        arrowprops={"arrowstyle": "-|>",
                                    "color": COLORS.get(method, "#777777"),
                                    "lw": 0.9})
            ax.text(20.9, yi - 0.18, f"{mean:.1f}", ha="right", va="top",
                    color=COLORS.get(method, "#777777"))
        else:
            ax.errorbar(mean, yi, xerr=[[mean - lo], [hi - mean]], fmt="o",
                        color=COLORS.get(method, "#777777"), capsize=1.8,
                        elinewidth=0.9)
    ax.axvline(0, color="#666666", linewidth=0.7, linestyle="--")
    ax.set_xlim(-3.5, 25.0)
    ax.set_yticks(y, [ALIASES[m] for m, *_ in rows], rotation=24,
                  va="center")
    ax.set_xlabel("CBS difference (pp) ↑", labelpad=1.5)
    panel(ax, "a", "Paired CBS")

    ax = axs[0, 1]
    for method in SIX:
        values = np.sort(routes[routes.Method == method]["min_clearance"])
        yy = np.arange(1, len(values) + 1) / len(values) * 100
        ax.step(values, yy, where="post", color=COLORS[method],
                label=ALIASES[method], linewidth=1.0)
    ax.axvline(1.35, color="#555555", linestyle="--", linewidth=0.7)
    ax.set_xlabel("Clearance (m) ↑", labelpad=1.5)
    ax.set_ylabel("ECDF (%)", labelpad=1.5)
    panel(ax, "b", "Clearance ECDF")

    ax = axs[1, 0]
    means = (runs[runs.Method.isin(SIX)].groupby("Method", sort=False).agg(
        SR=("SR", "mean"), SC=("SC", "mean"), CR=("CR", "mean"))
    )
    for method, row in means.iterrows():
        x, yv = row.SR * 100, row.SC
        size = 15 + 42 * (1 - row.CR)
        ax.scatter(x, yv, s=size, color=COLORS.get(method, "#777777"),
                   edgecolor="white", linewidth=0.45, zorder=3)
    ax.set_xlabel("SR (%) ↑", labelpad=1.5)
    ax.set_ylabel("SC (%) ↑", labelpad=1.5)
    panel(ax, "c", "SR--SC frontier")

    ax = axs[1, 1]
    data = []
    labels = []
    colors = []
    for method in SIX:
        values = routes[routes.Method == method]["mean_violation_depth"].fillna(0)
        data.append(values.to_numpy())
        labels.append(ALIASES[method])
        colors.append(COLORS[method])
    violin = ax.violinplot(data, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.65)
    violin["cmedians"].set_color("#222222"); violin["cmedians"].set_linewidth(0.8)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=54, ha="right")
    ax.set_ylabel("Violation depth (m) ↓", labelpad=1.5)
    panel(ax, "d", "Violation depth")

    handles = [mpl.lines.Line2D([], [], color=COLORS[m], label=ALIASES[m])
               for m in SIX]
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.52, 0.005), frameon=False,
               columnspacing=0.9, handlelength=1.7)
    save(fig, "fig5_common_plant.pdf", bottom=0.22, hspace=0.62, wspace=0.42)


def mismatch_figure():
    common = pd.read_csv(FINAL / "guarded_common_runs.csv")
    mismatch = pd.read_csv(FINAL / "guarded_mismatch_runs.csv")
    common["CBS"] = cbs(common); mismatch["CBS"] = cbs(mismatch)
    merged = common.merge(mismatch, on=["Method", "Seed"], suffixes=("_c", "_m"))
    merged["dCBS"] = merged.CBS_m - merged.CBS_c
    merged["dSC"] = merged.SC_m - merged.SC_c
    merged["dSR"] = (merged.SR_m - merged.SR_c) * 100
    merged["dCR"] = -(merged.CR_m - merged.CR_c) * 100
    merged["dClr"] = merged.MinClearance_m - merged.MinClearance_c
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 4.8))

    ax = axs[0, 0]
    selected = [m for m in SIX if m != "Scenario MPC"]
    x = np.array([0, 1])
    for method in selected:
        row_c = common[common.Method == method]
        row_m = mismatch[mismatch.Method == method]
        means = [cbs(row_c).mean(), cbs(row_m).mean()]
        ax.plot(x, means, marker="o", color=COLORS[method], label=ALIASES[method])
    ax.set_xticks(x, ["Common", "Heterog."])
    ax.set_ylabel("CBS (%) ↑", labelpad=1.5)
    panel(ax, "a", "CBS transfer")

    ax = axs[0, 1]
    rows = []
    for j, method in enumerate(SIX):
        vals = merged[merged.Method == method].dSC
        mean, lo, hi = percentile_ci(vals, seed=20261200 + j)
        rows.append((method, mean, lo, hi))
    rows.sort(key=lambda z: z[1])
    for yi, (method, mean, lo, hi) in enumerate(rows):
        ax.errorbar(mean, yi, xerr=[[mean-lo], [hi-mean]], fmt="o",
                    color=COLORS.get(method, "#777777"), capsize=1.5,
                    elinewidth=0.8)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=0.7)
    ax.set_yticks(range(len(rows)), [ALIASES[m] for m, *_ in rows],
                  rotation=24, va="center")
    ax.set_xlabel("SC change (pp) ↑", labelpad=1.5)
    panel(ax, "b", "SC change")

    ax = axs[1, 0]
    metrics = ["dSR", "dSC", "dCR", "dClr"]
    labels = ["SR (pp)", "SC (pp)", "−CR (pp)", "d min (m)"]
    matrix = np.array([[merged[merged.Method == m][metric].mean()
                        for metric in metrics] for m in SIX])
    scale = np.nanmax(np.abs(matrix), axis=0); scale[scale == 0] = 1
    shown = matrix / scale
    ax.imshow(shown, cmap=DIV_CMAP, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=62, ha="right")
    ax.set_yticks(range(len(SIX)), ["Fed." if m == "FedUMPC" else ALIASES[m] for m in SIX], rotation=20)
    for i in range(len(SIX)):
        for j in range(len(labels)):
            value = 0.0 if abs(matrix[i, j]) < 5e-4 else matrix[i, j]
            fmt = ".2f" if j == 3 else ".1f"
            ax.text(j, i, format(value, fmt), ha="center", va="center",
                    fontsize=8.5)
    panel(ax, "c", "Signed effects")

    ax = axs[1, 1]
    data = [merged[merged.Method == m].dCBS.to_numpy() for m in SIX]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops={"color": "#222222", "linewidth": 0.8})
    for patch, method in zip(bp["boxes"], SIX):
        patch.set_facecolor(COLORS[method]); patch.set_alpha(0.60)
        patch.set_edgecolor(COLORS[method])
    ax.axhline(0, color="#555555", linestyle="--", linewidth=0.7)
    ax.set_xticks(range(1, len(SIX)+1), [ALIASES[m] for m in SIX],
                  rotation=63, ha="right")
    ax.set_ylabel("CBS change (pp) ↑", labelpad=1.5)
    panel(ax, "d", "CBS spread")
    handles = [mpl.lines.Line2D([], [], color=COLORS[m], marker="o",
                                label=ALIASES[m]) for m in SIX]
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.52, 0.002), frameon=False,
               columnspacing=0.7, handlelength=1.3)
    save(fig, "fig6_heterogeneous_plant.pdf", bottom=0.23,
         hspace=0.65, wspace=0.64)


def federation_figure():
    data = pd.read_csv(AUX / "r1_fleet_scaling.csv")
    comm = pd.read_csv(AUX / "r1_communication_stress.csv")
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 4.5))
    order = ["Local Only", "FedAvg", "Prox-FL", "Centralized"]

    ax = axs[0, 0]
    for method in order:
        frame = data[data.Method == method]
        grouped = frame.groupby("N").Sparse_client_MSE.agg(["mean", "std"])
        ax.errorbar(grouped.index, grouped["mean"], yerr=grouped["std"],
                    marker="o", capsize=2, color=COLORS[method], label=method)
    ax.set_xticks([4, 16, 32])
    ax.set_xlabel("Clients, N (–)", labelpad=1.5)
    ax.set_ylabel("Data-poor NMSE (–) ↓", labelpad=1.5)
    panel(ax, "a", "Data scarcity")

    ax = axs[0, 1]
    shared = data[data.Final_drift.notna()]
    for method, frame in shared.groupby("Method"):
        ax.scatter(frame.Final_drift, frame.Mean_MSE, s=18,
                   color=COLORS[method], alpha=0.75, label=method)
    ax.set_xlabel("Client drift (–) ↓", labelpad=1.5)
    ax.set_ylabel("Fleet NMSE (–) ↓", labelpad=1.5)
    panel(ax, "b", "Drift and error")

    ax = axs[1, 0]
    methods = [m for m in order if data[data.Method == m].Empirical_B.notna().any()]
    matrix = np.array([[data[(data.Method == m) & (data.N == n)].Empirical_B.mean()
                        for n in [4, 16, 32]] for m in methods])
    image = ax.imshow(matrix, cmap=POS_CMAP, vmin=0, vmax=np.nanmax(matrix), aspect="auto")
    ax.set_xticks(range(3), ["N=4", "N=16", "N=32"])
    ax.set_yticks(range(len(methods)), ["FedAvg", "Prox-FL", "Central"], rotation=20)
    for i in range(len(methods)):
        for j in range(3):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=8.5)
    panel(ax, "c", "Dissimilarity")

    ax = axs[1, 1]
    group = comm.groupby("Scenario").agg(
        mse=("Mean_MSE", "mean"), traffic=("Round_traffic_MB", "mean"))
    labels = ["Full", "Half", "Stale"]
    names = ["Synchronous full", "50% participation", "50% + two-round stale"]
    group = group.loc[names]
    xx = np.arange(3)
    bars = ax.bar(xx, group.traffic, color=PALETTE[:3], alpha=0.72)
    ax.set_ylabel("Payload (MB) ↓", labelpad=1.5)
    ax.set_xticks(xx, labels, rotation=32, ha="right")
    ax2 = ax.twinx()
    ax2.plot(xx, group.mse, color=FED, marker="o", linewidth=1.1)
    ax2.set_ylabel("NMSE (–) ↓", labelpad=1.5)
    ax2.spines["top"].set_visible(False)
    panel(ax, "d", "Participation")
    handles = [mpl.lines.Line2D([], [], color=COLORS[m], marker="o", label=m)
               for m in order]
    fig.legend(handles=handles, ncol=4, loc="lower center",
               bbox_to_anchor=(0.50, 0.035), frameon=False,
               columnspacing=0.55, handlelength=1.0, handletextpad=0.35)
    save(fig, "fig7_federation.pdf", bottom=0.19, hspace=0.65, wspace=0.70)


def robustness_figure():
    summary = pd.read_csv(artifact_path("results_r1_wind_only") / "six_condition_summary.csv")
    runs = pd.read_csv(artifact_path("results_r1_wind_only") / "six_condition_runs.csv")
    conditions = list(summary.Condition.drop_duplicates())
    x = np.arange(len(conditions))
    fig, axs = plt.subplots(2, 3, figsize=(3.5, 4.8))

    for ax, metric, ylabel, letter, title in [
        (axs[0, 0], "SR", "SR (%) ↑", "a", "Success"),
        (axs[0, 1], "SC", "SC (%) ↑", "b", "Compliance"),
        (axs[0, 2], "MinClearance", "Clearance (m) ↑", "c", "Clearance"),
    ]:
        for method in SIX:
            frame = summary[summary.Method == method].set_index("Condition").loc[conditions]
            mean = frame[f"{metric}_mean"].to_numpy()
            lo = frame[f"{metric}_lo"].to_numpy()
            hi = frame[f"{metric}_hi"].to_numpy()
            ax.plot(x, mean, marker="o", color=COLORS[method], label=ALIASES[method])
            ax.fill_between(x, lo, hi,
                            color=COLORS[method], alpha=0.08, linewidth=0)
        ax.set_xticks(x, ["Nom.", "N.10", "N.25", "B.50", "Cmp.", "Wind"],
                      rotation=65, ha="right")
        ax.set_ylabel(ylabel, labelpad=1.0)
        if metric == 'MinClearance':
            ax.set_yticks([-0.5, 0.0, 0.5], ['−0.5', '0', '0.5'])
        panel(ax, letter, title)

    ax = axs[1, 0]
    contact = (runs.groupby(["Condition", "Method"], sort=False).CR
               .agg(["mean", "std"]).reset_index())
    width = 0.12
    for j, method in enumerate(SIX):
        frame = contact[contact.Method == method].set_index("Condition").loc[conditions]
        ax.bar(x + (j - 2.5) * width, frame["mean"] * 4,
               width=width, color=COLORS[method], alpha=0.72)
    ax.set_xticks(x, ["Nom.", "N.10", "N.25", "B.50", "Cmp.", "Wind"],
                  rotation=65, ha="right")
    ax.set_ylabel("Contacts (–) ↓", labelpad=1.0)
    panel(ax, "d", "Contacts")

    ax = axs[1, 1]
    base = summary[summary.Condition == conditions[0]].set_index("Method")
    matrix = []
    for method in SIX:
        frame = summary[summary.Method == method].set_index("Condition").loc[conditions]
        signed = np.column_stack([
            frame.SR_mean - base.loc[method, "SR_mean"],
            frame.SC_mean - base.loc[method, "SC_mean"],
            -(frame.CR_mean - base.loc[method, "CR_mean"]),
        ])
        matrix.append(np.min(signed[1:], axis=0))
    matrix = np.asarray(matrix)
    vmax = max(1.0, float(np.max(np.abs(matrix))))
    image = ax.imshow(matrix, cmap=DIV_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(3), ["SR", "SC", "−CR"], rotation=45, ha="right")
    ax.set_yticks(range(len(SIX)),
                  ["Nom.", "Tube", "Scen.", "GP", "Fixed", "Fed."],
                  rotation=24)
    for i in range(len(SIX)):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center",
                    fontsize=8.5)
    panel(ax, "e", "Worst loss")

    ax = axs[1, 2]
    for method in SIX:
        frame = summary[summary.Method == method]
        avg = 0.5 * (frame.SR_mean.mean() + frame.SC_mean.mean())
        worst = np.minimum(frame.SR_mean, frame.SC_mean).min()
        ax.scatter(avg, worst, s=22, color=COLORS[method],
                   edgecolor="white", linewidth=0.4, zorder=3)
    ax.set_xlabel("Mean utility (%) ↑", labelpad=1.0)
    ax.set_ylabel("Worst endpoint (%) ↑", labelpad=1.0)
    panel(ax, "f", "Trade-off")

    handles = [mpl.lines.Line2D([], [], color=COLORS[m], marker="o",
                                label=ALIASES[m]) for m in SIX]
    fig.legend(handles=handles, ncol=3, loc="lower center",
               bbox_to_anchor=(0.52, 0.002), frameon=False,
               columnspacing=0.7, handlelength=1.3)
    save(fig, "fig8_robustness.pdf", bottom=0.19, hspace=0.70, wspace=0.85)


def deployment_figure():
    mc = pd.read_csv(FINAL / "r1_mc_validation_summary.csv")
    latency = pd.read_csv(FINAL / "r1_candidate_latency.csv")
    sensitivity = pd.read_csv(FINAL / "r1_obstacle_sensitivity.csv")
    mc_control = pd.read_csv(FINAL / "r1_mc_closed_loop_summary.csv")
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 4.5))

    ax = axs[0, 0]
    query = pd.read_csv(FINAL / "r1_mc_query_diagnostics.csv")
    ax.plot(query.M, query.Mean_query_repeat_variance, marker="o", color=FED)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(query.M, [str(int(x)) for x in query.M])
    ax.set_xlabel("MC samples, M", labelpad=1.5)
    ax.set_ylabel("Query variance (–) ↓", labelpad=1.5)
    panel(ax, "a", "Repeatability")

    ax = axs[0, 1]
    mc_colors = [PALETTE[0], PALETTE[1], PALETTE[2], FED]
    for (m, frame), color in zip(latency.groupby("M"), mc_colors):
        ax.plot(frame.Candidates, frame.Mean_ms, marker="o",
                label=f"M={m}", color=color)
        ax.fill_between(frame.Candidates, frame.Mean_ms, frame.P95_ms,
                        color=color, alpha=0.10)
    ax.set_xticks([4, 8, 12, 16])
    ax.set_ylim(25, 76)
    ax.set_xlabel("Candidate count", labelpad=1.5)
    ax.set_ylabel("Scoring time (ms) ↓", labelpad=1.5)
    panel(ax, "b", "Scoring time")

    ax = axs[1, 0]
    sensitivity["ratio"] = sensitivity.Value / sensitivity.Base
    means = (sensitivity.groupby(["Parameter", "ratio"]).SC.mean()
             .reset_index())
    base = means.iloc[(means.ratio - 1).abs().groupby(means.Parameter).idxmin()]
    base_map = base.set_index("Parameter").SC
    means["delta"] = means.apply(
        lambda row: row.SC - base_map.loc[row.Parameter], axis=1)
    parameters = list(means.Parameter.drop_duplicates())
    for yi, parameter in enumerate(parameters):
        frame = means[means.Parameter == parameter]
        ax.hlines(yi, frame.delta.min(), frame.delta.max(), color="#777777",
                  linewidth=1.0)
        colors = ["#FF4F4F" if value >= 0 else "#007FFF"
                  for value in frame.delta]
        ax.scatter(frame.delta, np.full(len(frame), yi), c=colors, s=15,
                   edgecolor="white", linewidth=0.35, zorder=3)
    ax.axvline(0, color="#555555", linestyle="--", linewidth=0.7)
    parameter_labels = {
        "q_obs": r"$q_{\mathrm{obs}}$",
        "inner_a": r"$c_{\mathrm{in}}$",
        "d_inf": r"$d_{\mathrm{inf}}$",
        "c_s": r"$c_s$",
    }
    ax.set_yticks(range(len(parameters)),
                  [parameter_labels.get(p, p) for p in parameters], rotation=22)
    ax.set_xlabel("SC change (pp) ↑", labelpad=1.5)
    panel(ax, "c", "Sensitivity")

    ax = axs[1, 1]
    x = np.arange(len(mc_control))
    ax.bar(x, mc_control.SR_pct, color=PALETTE[0], alpha=0.72,
           label="SR")
    ax.set_xticks(x, [f"M={int(v)}" for v in mc_control.M],
                  rotation=30, ha="right")
    ax.set_ylabel("SR (%) ↑", labelpad=1.5)
    ax2 = ax.twinx(); ax2.plot(x, mc_control.SC_pct, color=FED,
                               marker="o", label="SC")
    ax2.set_ylabel("SC (%) ↑", labelpad=1.5)
    ax2.spines["top"].set_visible(False)
    panel(ax, "d", "MC control")
    handles = [mpl.lines.Line2D([], [], color=color, marker="o", label=f"M={m}")
               for (m, _), color in zip(latency.groupby("M"), mc_colors)]
    fig.legend(handles=handles, ncol=2, loc="lower center",
               bbox_to_anchor=(0.52, 0.005), frameon=False,
               columnspacing=0.8, handlelength=1.4)
    save(fig, "fig9_deployment_sensitivity.pdf", bottom=0.19,
         hspace=0.65, wspace=0.70)


def uncertainty_figure():
    samples = pd.read_csv(AUX / "r1_uncertainty_samples.csv")
    summary = pd.read_csv(AUX / "r1_uncertainty_summary.csv")
    preferred = ["Nominal", "High noise", "Mass shift", "Wind", "Pooled"]
    summary = (summary.assign(
        _order=pd.Categorical(summary.Scenario, preferred, ordered=True))
        .sort_values("_order").drop(columns="_order"))
    fig, axs = plt.subplots(2, 2, figsize=(3.5, 4.6))

    ax = axs[0, 0]
    hb = ax.hexbin(samples.Uncertainty, samples.Prediction_error, gridsize=32,
                   mincnt=1, cmap=POS_CMAP)
    ax.set_xlabel("Uncertainty U (–)", labelpad=1.5)
    ax.set_ylabel("Normalized error ↓", labelpad=1.5)
    panel(ax, "a", "Error density")
    cax = ax.inset_axes([0.54, 0.84, 0.40, 0.055])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8.5, pad=1, length=2)
    cb.locator = mpl.ticker.MaxNLocator(nbins=2)
    cb.update_ticks()

    ax = axs[0, 1]
    quartiles = pd.qcut(samples.Uncertainty, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    data = [samples.loc[quartiles == q, "Prediction_error"].to_numpy()
            for q in ["Q1", "Q2", "Q3", "Q4"]]
    violin = ax.violinplot(data, showmedians=True, showextrema=False)
    for body, color in zip(violin["bodies"], [PALETTE[0], PALETTE[1], PALETTE[2], FED]):
        body.set_facecolor(color); body.set_edgecolor(color); body.set_alpha(0.68)
    violin["cmedians"].set_color("#222222"); violin["cmedians"].set_linewidth(0.8)
    ax.set_xticks(range(1, 5), ["Q1", "Q2", "Q3", "Q4"])
    ax.set_xlabel("Uncertainty quartile", labelpad=1.5)
    ax.set_ylabel("Normalized error ↓", labelpad=1.5)
    panel(ax, "b", "Error violins")

    ax = axs[1, 0]
    for q, values, color in zip(["Q1", "Q2", "Q3", "Q4"], data,
                                [PALETTE[0], PALETTE[1], PALETTE[2], FED]):
        values = np.sort(values); yy = np.arange(1, len(values)+1) / len(values) * 100
        ax.step(values, yy, where="post", label=q, color=color)
    ax.set_xlabel("Error (–) ↓", labelpad=1.5)
    ax.set_ylabel("ECDF (%)", labelpad=1.5)
    panel(ax, "c", "Error ECDFs")

    ax = axs[1, 1]
    x = np.arange(len(summary))
    ax.bar(x, summary.False_low_uncertainty_pct, color=PALETTE[:len(summary)], alpha=0.65)
    ax.set_ylabel("False-low frequency (%) ↓", labelpad=1.5)
    ax.set_xticks(x, ["Nom.", "Noise", "Mass", "Wind", "Pool"], rotation=55, ha="right")
    ax2 = ax.twinx(); ax2.plot(x, summary.Spearman_rho, color=FED, marker="o")
    ax2.set_ylabel("Spearman correlation ↑", labelpad=1.5)
    ax2.spines["top"].set_visible(False)
    panel(ax, "d", "Scenario reliability")
    handles = [mpl.lines.Line2D([], [], color=c, label=q)
               for q, c in zip(["Q1", "Q2", "Q3", "Q4"],
                               [PALETTE[0], PALETTE[1], PALETTE[2], FED])]
    fig.legend(handles=handles, ncol=4, loc="lower center",
               bbox_to_anchor=(0.52, 0.035), frameon=False,
               columnspacing=0.75, handlelength=1.25)
    save(fig, "fig10_uncertainty_error.pdf", bottom=0.18, hspace=0.57, wspace=0.60)


def main():
    setup_style()
    common_plant_figure()
    mismatch_figure()
    federation_figure()
    robustness_figure()
    deployment_figure()
    uncertainty_figure()


if __name__ == "__main__":
    main()
