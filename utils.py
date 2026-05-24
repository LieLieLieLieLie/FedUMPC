# utils.py
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
try:
    import matplotlib.patches as patches
    import matplotlib.patches as mpatches  # alias used in plot_robustness_bars
    from matplotlib.lines import Line2D
except Exception:
    patches = mpatches = Line2D = None
try:
    import pandas as pd
except Exception:
    pd = None
import os
try:
    import seaborn as sns
except Exception:
    sns = None
from config import Config
try:
    import matplotlib.font_manager as fm
except Exception:
    fm = None
import torch

# 通用字体设置
if fm is not None:
    font_new_roman = fm.FontProperties(family='Times New Roman', style='normal', size=28)
    font_bold = fm.FontProperties(family='Times New Roman', weight='bold', size=30)
    font_legend = fm.FontProperties(family='Times New Roman', style='normal', size=26)
else:
    font_new_roman = font_bold = font_legend = None


class VehicleModel:
    def __init__(self, mass=Config.MASS, drag=Config.DRAG):
        self.L = Config.L
        self.dt = Config.DT
        self.mass = mass
        self.drag = drag

    def step(self, state, control, noise_std=0.0, disturbance=0.0):
        x, y, v, yaw = state
        steer, accel = control
        steer = np.clip(steer, -Config.MAX_STEER, Config.MAX_STEER)
        accel = np.clip(accel, -Config.MAX_ACCEL, Config.MAX_ACCEL)
        beta = np.arctan(0.5 * np.tan(steer))
        dx = v * np.cos(yaw + beta) + np.random.normal(0, noise_std)
        dy = v * np.sin(yaw + beta) + np.random.normal(0, noise_std)
        dv = (accel - self.drag * v + disturbance)
        dyaw = (v / self.L) * np.sin(beta)
        new_x = x + dx * self.dt
        new_y = y + dy * self.dt
        new_v = max(0.0, v + dv * self.dt)
        new_yaw = yaw + dyaw * self.dt
        return np.array([new_x, new_y, new_v, new_yaw])


class CrossingEnv:
    def __init__(self, seed=None, obstacle_density=None):
        if seed is not None: np.random.seed(seed)
        self.obstacle_density = obstacle_density if obstacle_density is not None else Config.OBSTACLE_DENSITY
        self.obstacles = []
        self._gen_obstacles()

    def _gen_obstacles(self):
        # Critical path trap obstacles (fixed positions)
        self.obstacles.append({'x': -9.0, 'y': -0.5, 'r': 2.2})
        self.obstacles.append({'x': 9.0, 'y': 0.5, 'r': 2.2})
        self.obstacles.append({'x': -0.5, 'y': -9.0, 'r': 2.2})
        self.obstacles.append({'x': 0.5, 'y': 9.0, 'r': 2.2})
        self.obstacles.append({'x': 0.0, 'y': 0.0, 'r': 2.5})

        density = self.obstacle_density
        for _ in range(int(8 * density)):
            x = np.random.uniform(-16, 16)
            y = np.random.uniform(-16, 16)
            if np.linalg.norm([x, y]) < 8.0: continue
            if abs(x) > 15 and abs(y) > 15: continue
            self.obstacles.append({'x': x, 'y': y, 'r': np.random.uniform(1.2, 1.8)})

    def get_min_dist(self, x, y):
        min_d = float('inf')
        for obs in self.obstacles:
            d = np.sqrt((x - obs['x']) ** 2 + (y - obs['y']) ** 2) - obs['r']
            if d < min_d: min_d = d
        return min_d


# --- 绘图函数 ---

def plot_crossing_comparison(env, all_results, targets, starts, filename):
    if not filename.endswith('.pdf'): filename = filename.replace('.png', '.pdf')

    font_title_huge = fm.FontProperties(family='Times New Roman', weight='bold', size=52)
    font_tick_huge = fm.FontProperties(family='Times New Roman', style='normal', size=42)
    font_legend_huge = fm.FontProperties(family='Times New Roman', weight='bold', size=42)

    fig, axs = plt.subplots(1, 4, figsize=(48, 12))
    axs = axs.flatten()

    method_styles = {
        'Linear MPC': {'ls': ':', 'lw': 8.0, 'z': 2},
        'Robust MPC': {'ls': '--', 'lw': 8.0, 'z': 3},
        'FedRMPC': {'ls': '-', 'lw': 10.0, 'z': 5}
    }

    for i in range(Config.NUM_AGENTS):
        ax = axs[i]
        for obs in env.obstacles:
            ax.add_patch(patches.Circle((obs['x'], obs['y']), obs['r'], color='#D2B48C', alpha=0.6, zorder=1))

        for method_name, style in method_styles.items():
            if method_name in all_results:
                res = all_results[method_name][i]
                t = np.array(res['traj'])
                if len(t) > 0:
                    ax.plot(t[:, 0], t[:, 1], color=Config.COLORS[method_name],
                            ls=style['ls'], lw=style['lw'], alpha=0.9, zorder=style['z'])
                    if not res['success']:
                        ax.scatter(t[-1, 0], t[-1, 1], marker='x', color='#8B0000', s=800, zorder=25, lw=8)

        ax.scatter(starts[i][0], starts[i][1], marker='s', color='k', s=400, zorder=20)
        ax.scatter(targets[i][0], targets[i][1], marker='*', color='#FFD700', edgecolors='k', s=1000, zorder=20)

        ax.set_title(f"Agent {i + 1}", fontproperties=font_title_huge, pad=30)
        ax.set_xlim(-23, 23);
        ax.set_ylim(-23, 23)
        ax.grid(True, ls=':', alpha=0.5, color='gray', linewidth=2)
        ax.set_aspect('equal')

        ax.tick_params(axis='both', colors='black', labelsize=42, width=4, length=12, pad=10)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(font_tick_huge)

    lines = [
        Line2D([0], [0], color=Config.COLORS['Linear MPC'], lw=8.0, ls=':', label='Linear MPC'),
        Line2D([0], [0], color=Config.COLORS['Robust MPC'], lw=8.0, ls='--', label='Robust MPC'),
        Line2D([0], [0], color=Config.COLORS['FedRMPC'], lw=10.0, ls='-', label='FedRMPC'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='k', markersize=24, label='Start'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700', markeredgecolor='k', markersize=36,
               label='Goal'),
        Line2D([0], [0], marker='x', color='#8B0000', markerfacecolor='#8B0000', markersize=24, lw=6, label='Collision')
    ]
    fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=6, frameon=False,
               prop=font_legend_huge)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Colour/style helpers
# ─────────────────────────────────────────────────────────────────────────────
def _c(name):  return Config.COLORS.get(name, '#888888')
def _m(name):  return Config.MARKERS.get(name, 'o')
def _ls(name): return Config.LINESTYLES.get(name, '-')


# ─────────────────────────────────────────────────────────────────────────────
#  Trajectory comparison (N methods side by side, one agent panel each)
# ─────────────────────────────────────────────────────────────────────────────
def plot_crossing_comparison(env, all_trajs, targets, starts, filename):
    """
    Trajectory comparison across methods.
    Fix: always draws whatever trajectory exists (even 1-step ones).
    Annotates each agent with success/collision/timeout status.
    """
    if not filename.endswith('.pdf'):
        filename = filename.replace('.png', '.pdf')
    methods = list(all_trajs.keys())
    n_m = len(methods)
    fig, axs = plt.subplots(1, n_m, figsize=(10*n_m, 10))
    if n_m == 1:
        axs = [axs]

    agent_colors = ['#CC3333', '#3366CC', '#228B22', '#CC8800']

    for ax, mname in zip(axs, methods):
        # background
        ax.set_facecolor('#F8F8F8')
        # obstacles
        for obs in env.obstacles:
            ax.add_patch(plt.Circle((obs['x'], obs['y']), obs['r'],
                                    color='#BBBBBB', zorder=1))
            ax.add_patch(plt.Circle((obs['x'], obs['y']), obs['r'],
                                    fill=False, edgecolor='#666666',
                                    linewidth=2, zorder=2))
        trajs = all_trajs[mname]
        n_succ = sum(1 for t in trajs.values() if t.get('success', False))
        n_coll = sum(1 for t in trajs.values() if t.get('collision', False))

        for i in range(4):
            t_data = trajs[i]
            traj   = np.array(t_data['traj'])   # at least 2 points (start + 1)
            col    = agent_colors[i]
            # trajectory line
            ax.plot(traj[:, 0], traj[:, 1], color=col, lw=3.5,
                    zorder=3, alpha=0.90, solid_capstyle='round')
            # direction arrow at mid-point
            if len(traj) >= 4:
                mid = len(traj) // 2
                dx  = traj[mid,0] - traj[mid-1,0]
                dy  = traj[mid,1] - traj[mid-1,1]
                if dx**2 + dy**2 > 1e-6:
                    ax.annotate('', xy=(traj[mid,0], traj[mid,1]),
                                xytext=(traj[mid-1,0], traj[mid-1,1]),
                                arrowprops=dict(arrowstyle='->', color=col,
                                                lw=2.5), zorder=6)
            # start marker
            ax.plot(starts[i][0], starts[i][1], 'o', color=col,
                    ms=12, zorder=7, markeredgecolor='white', markeredgewidth=2)
            # target marker
            ax.plot(targets[i][0], targets[i][1], '*', color=col,
                    ms=16, zorder=7, markeredgecolor='white', markeredgewidth=1)
            # end marker: ✓ or ✗
            end = traj[-1]
            if t_data.get('success', False):
                ax.plot(end[0], end[1], 's', color=col, ms=11, zorder=8,
                        markeredgecolor='white', markeredgewidth=1.5)
            elif t_data.get('collision', False):
                ax.plot(end[0], end[1], 'X', color='red', ms=13, zorder=8,
                        markeredgecolor='white', markeredgewidth=1.5)

        title_str = f"{mname}\n({n_succ}/4 reached, {n_coll} collisions)"
        ax.set_title(title_str, fontproperties=font_bold, fontsize=36,
                     color='black', pad=14)
        ax.set_xlim(-23, 23); ax.set_ylim(-23, 23)
        ax.set_aspect('equal')
        ax.set_xlabel('X (m)', fontproperties=font_bold, fontsize=34)
        ax.set_ylabel('Y (m)', fontproperties=font_bold, fontsize=34)
        ax.tick_params(labelsize=30)
        ax.grid(True, ls=':', alpha=0.35, color='gray')

    plt.tight_layout(pad=1.5)
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), dpi=300,
                bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Grouped bar chart (replaces radar) with line overlay for key metrics
# ─────────────────────────────────────────────────────────────────────────────
def plot_radar_chart_7(scenario_results, filename):
    """
    Replaces the radar/spider chart with a clean grouped bar + line figure.

    Layout: 2-row × 3-col subplot grid
      Row 1: Success Rate (bar) | Safety Compliance (bar) | Collision-Free Rate (bar)
      Row 2: Safety Margin (bar+line) | Time Efficiency (bar) | Uncertainty Util. (bar)

    Each bar group = one method, coloured by Config.COLORS.
    Compute(ms) is intentionally excluded from paper-facing metrics.
    """
    if not filename.endswith('.pdf'):
        filename = filename.replace('.png', '.pdf')

    methods = [m for m in Config.COLORS if m in scenario_results]
    if not methods:
        return

    # (key_mean, key_std, ylabel, scale, higher_is_better)
    panels = [
        ('SR_mean',      'SR_std',       'Success Rate (%)',         100,   True),
        ('SC_mean',      'SC_std',       'Safety Compliance (%)',    1,     True),
        ('CR_mean',      'CR_std',       'Collision-Free Rate (%)',  100,   True),
        ('AvgDist_mean', 'AvgDist_std',  'Avg Safety Margin (m)',    1,     True),
        ('TE_mean',      'TE_std',       'Time Efficiency (%)',      100,   True),
        ('UU_mean',      'UU_std',       'Uncertainty Util. (%)',    100,   True),
    ]

    fig, axs = plt.subplots(2, 3, figsize=(20, 12.5))
    axs = axs.flatten()

    x  = np.arange(len(methods))
    bw = 0.65

    for ax, (km, ks, ylabel, scale, hib) in zip(axs, panels):
        vals = []
        errs = []
        for m in methods:
            agg = scenario_results[m]
            if km == 'CR_mean':
                v = (1.0 - agg.get(km, 0.0)) * scale
            else:
                v = agg.get(km, 0.0) * scale
            if not hib:
                v = -v   # already negated above for CR
            e = agg.get(ks, 0.0) * abs(scale)
            vals.append(v); errs.append(e)

        colors = [_c(m) for m in methods]
        bars = ax.bar(x, vals, bw, color=colors, zorder=3,
                      edgecolor='white', linewidth=1.2)
        ax.errorbar(x, vals, yerr=errs, fmt='none',
                    ecolor='#333333', elinewidth=2, capsize=5, zorder=4)

        # line overlay connecting bar tops
        ax.plot(x, vals, 'o--', color='#444444', lw=2.5,
                ms=8, zorder=5, alpha=0.7)

        ax.set_xticks([])
        ax.set_ylabel(ylabel, fontproperties=font_bold, fontsize=20)
        ax.set_title(ylabel, fontproperties=font_bold, fontsize=20, color='black', pad=6)
        ax.grid(True, axis='y', ls=':', alpha=0.55, color='gray', zorder=0)
        ax.tick_params(axis='y', labelsize=18)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # annotate bar tops
        for bar, v in zip(bars, vals):
            if abs(v) > 0.5:
                label = f'{v:.0f}' if abs(v) >= 5 else f'{v:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        label, ha='center', va='bottom', fontsize=14,
                        fontproperties=font_new_roman, color='#222222')

    # legend directly below subplots (not far from bars)
    handles = [mpatches.Patch(color=_c(m), label=m) for m in methods]
    ncol = min(len(methods), 6)
    fig.legend(handles=handles, loc='lower center', ncol=ncol,
               frameon=False, prop=font_new_roman, fontsize=18,
               bbox_to_anchor=(0.5, -0.005))

    plt.tight_layout(rect=[0, 0.07, 1, 1], pad=0.6, w_pad=1.0, h_pad=1.0)
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename),
                bbox_inches='tight', dpi=300)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  FL convergence curves (MSE + Drift, two subplots)
# ─────────────────────────────────────────────────────────────────────────────
def plot_fl_curves(hist, methods, mse_file, drift_file):
    rounds = hist['Round']
    PROPOSED = {'FedProx', 'DP-FedProx'}   # 加粗高亮

    # ── MSE ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 9))
    for m in methods:
        key = f'{m}_MSE'
        if key not in hist: continue
        vals = np.array(hist[key], dtype=float)
        if np.isnan(vals).all(): continue          # 跳过Scaffold爆炸
        lw   = 6.5 if m in PROPOSED else 4.0
        ms   = 13  if m in PROPOSED else 10
        zord = 5   if m in PROPOSED else 3
        lbl  = f'{m} (proposed)' if m in PROPOSED else m
        ax.plot(rounds, vals.tolist(), color=_c(m), lw=lw,
                ls=_ls(m), marker=_m(m), markersize=ms, zorder=zord,
                markevery=max(1, len(rounds)//8), label=lbl)
    # annotate FedProx
    last_r = list(rounds)[-1]
    fp_vals = hist.get('FedProx_MSE', [])
    if fp_vals and not np.isnan(float(fp_vals[-1]) if fp_vals else float('nan')):
        try:
            ax.annotate('Best federated\n(lowest drift)',
                        xy=(last_r, float(fp_vals[-1])),
                        xytext=(last_r - max(1, len(rounds)//4), float(fp_vals[-1]) * 3.5),
                        fontsize=22, color=_c('FedProx'),
                        fontproperties=font_new_roman,
                        arrowprops=dict(arrowstyle='->', color=_c('FedProx'), lw=2))
        except Exception:
            pass
    ax.set_xlabel('Communication Round', fontproperties=font_bold, fontsize=28)
    ax.set_ylabel('Validation MSE',      fontproperties=font_bold, fontsize=28)
    # y축 클리핑: Scaffold 등 극단값 제외하고 FedProx 가시성 확보
    all_vis = [v for k,v in hist.items() if k.endswith('_MSE')
               for v in ([v] if not hasattr(v,'__iter__') else v)
               if isinstance(v,(int,float)) and not np.isnan(v) and v < 1.0]
    if all_vis:
        ax.set_ylim(0, min(max(all_vis)*1.25, 0.25))
    # 주석: FedProx의 낮은 drift 우위 강조
    handles, labels = ax.get_legend_handles_labels()
    ncol_lg = min(len(handles), 4)
    fig.legend(handles, labels, loc='upper center', ncol=ncol_lg,
               frameon=False, prop=font_new_roman, fontsize=20,
               bbox_to_anchor=(0.5, 1.01))
    ax.grid(True, ls=':', alpha=0.55, color='gray')
    ax.tick_params(labelsize=24, width=2.5, length=7)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(os.path.join(Config.RESULTS_DIR, mse_file), dpi=300, bbox_inches='tight')
    plt.close()

    # ── Drift (exclude Local Only; clip Scaffold爆炸值) ──────────────────────
    PROPOSED = {'FedProx', 'DP-FedProx'}
    drift_methods = [m for m in methods if m != 'Local Only']
    last_r = list(rounds)[-1]
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axhline(1.0, color='#444444', ls=':', lw=3.0, alpha=0.85, zorder=1)
    ax.text(list(rounds)[0], 1.03, 'Low drift threshold (stable)',
            fontsize=18, color='#444444', fontproperties=font_new_roman,
            va='bottom')
    for m in drift_methods:
        key = f'{m}_Drift'
        if key not in hist: continue
        vals = np.array(hist[key], dtype=float)
        nans = np.isnan(vals)
        if nans.all(): continue
        if nans.any():
            ok_idx = np.where(~nans)[0]
            vals[nans] = np.interp(np.where(nans)[0], ok_idx, vals[ok_idx])
        # 硬上限5.0防止Scaffold(1234)撑开y轴; 同时clip p95
        p95  = np.nanpercentile(vals, 95)
        vals = np.clip(vals, 0, min(p95 * 1.5, 5.0))
        lw   = 6.5 if m in PROPOSED else 4.0
        ms   = 13  if m in PROPOSED else 10
        zord = 5   if m in PROPOSED else 3
        lbl  = f'{m} (proposed)' if m in PROPOSED else m
        ax.plot(rounds, vals.tolist(), color=_c(m), lw=lw,
                ls=_ls(m), marker=_m(m), markersize=ms, zorder=zord,
                markevery=max(1, len(rounds)//8), label=lbl)
    try:
        ax.annotate('Stable region',
                    xy=(last_r, 1.0),
                    xytext=(last_r - max(1, len(rounds)//3), 1.45),
                    fontsize=22, color='#444444',
                    fontproperties=font_new_roman,
                    arrowprops=dict(arrowstyle='->', color='#444444', lw=2))
    except Exception:
        pass
    ax.set_xlabel('Communication Round', fontproperties=font_bold, fontsize=28)
    ax.set_ylabel('Client Drift (L2)',   fontproperties=font_bold, fontsize=28)
    handles, labels = ax.get_legend_handles_labels()
    ncol_lg = min(len(handles), 4)
    fig.legend(handles, labels, loc='upper center', ncol=ncol_lg,
               frameon=False, prop=font_new_roman, fontsize=20,
               bbox_to_anchor=(0.5, 1.01))
    ax.grid(True, ls=':', alpha=0.55, color='gray')
    ax.tick_params(labelsize=24, width=2.5, length=7)
    plt.tight_layout(rect=[0, 0, 1, 0.88])
    plt.savefig(os.path.join(Config.RESULTS_DIR, drift_file), dpi=300, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Robustness bar chart (3 metrics × 4 scenarios × 6 methods)
# ─────────────────────────────────────────────────────────────────────────────
def plot_robustness_bars(df, filename):
    """
    v6: 각 서브플롯에서 최고 성능 알고리즘 표시 (★ 마커 + 굵은 테두리)
    """
    if not filename.endswith('.pdf'):
        filename = filename.replace('.png', '.pdf')

    import matplotlib.colors as mcolors

    def _darken(color, factor=0.78):
        rgb = np.array(mcolors.to_rgb(color))
        return tuple(np.clip(rgb * factor, 0.0, 1.0))

    scenarios = df['Scenario'].unique()
    methods   = df['Method'].unique()
    metrics   = [('SR (%)', 'Success Rate (%)'),
                 ('SC (%)', 'Safety Compliance (%)'),
                 ('AvgDist', 'Avg Safety Margin (m)')]
    n_sc  = len(scenarios)
    n_met = len(metrics)

    fig, axs = plt.subplots(n_met, n_sc, figsize=(7*n_sc, 6*n_met))
    if n_sc == 1: axs = axs[:, np.newaxis]

    bw = 0.8 / max(len(methods), 1)
    xs = np.arange(1)

    for row, (col_key, ylabel) in enumerate(metrics):
        for col, sc in enumerate(scenarios):
            ax = axs[row][col]
            sub = df[df['Scenario'] == sc]
            # 최고값 찾기
            best_val = max(float(sub[col_key].values[k]) for k in range(len(sub)))
            for j, m in enumerate(methods):
                msub = sub[sub['Method'] == m]
                if msub.empty: continue
                val = float(msub[col_key].values[0])
                offset = (j - len(methods)/2 + 0.5) * bw
                is_best = abs(val - best_val) < 0.5
                face = _darken(_c(m)) if is_best else _c(m)
                ax.bar(xs + offset, [val], bw*0.88,
                       color=face, label=m, zorder=3,
                       edgecolor='white', linewidth=0.8,
                       hatch='//////' if is_best else None)
                if is_best:
                    ax.bar(xs + offset, [val], bw*0.88,
                           color='none', zorder=4,
                           edgecolor='#222222', linewidth=1.4)
            ax.set_title(sc if row==0 else '', fontproperties=font_bold,
                         fontsize=22)
            ax.set_ylabel(ylabel if col==0 else '',
                          fontproperties=font_bold, fontsize=20)
            ax.set_xticks([])
            ax.grid(True, axis='y', ls=':', alpha=0.5, color='gray')
            ax.tick_params(labelsize=18)

    # Legend at bottom in a single row
    handles = [mpatches.Patch(color=_c(m), label=m) for m in methods]
    handles.append(mpatches.Patch(facecolor='#777777', edgecolor='white',
                                  hatch='//////', label='Best in panel'))
    fig.legend(handles=handles, loc='lower center', ncol=min(len(handles), 7),
               frameon=False, prop=font_new_roman, fontsize=16,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), dpi=300,
                bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Ablation plots
# ─────────────────────────────────────────────────────────────────────────────
def plot_ablation_reliability(df, filename):
    """
    消融实验可靠性图: SR(%) 柱状图 + SC(%) 折线图 (双轴)
    只保留2个图例: Success Rate (bar) 和 Safety Compliance (line)
    """
    if not filename.endswith('.pdf'):
        filename = filename.replace('.png', '.pdf')
    cfgs   = list(df['Config'].unique())
    x      = np.arange(len(cfgs))
    bw     = 0.55
    colors = [_c(c) for c in cfgs]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    sr_vals = [float(df[df['Config']==c]['SR (%)'].mean())  for c in cfgs]
    sc_vals = [float(df[df['Config']==c]['SC (%)'].mean())  for c in cfgs]

    # Bars for SR
    bars = ax1.bar(x, sr_vals, bw, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5, zorder=3)
    for bar, v in zip(bars, sr_vals):
        if v > 2:
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                     f'{v:.1f}', ha='center', va='bottom', fontsize=22,
                     fontproperties=font_bold)

    ax1.set_ylabel('Success Rate (%)', fontproperties=font_bold, fontsize=24)
    ax1.set_ylim(0, max(sr_vals) * 1.35)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfgs, fontproperties=font_new_roman, fontsize=22,
                         rotation=10, ha='right')
    ax1.grid(True, axis='y', ls=':', alpha=0.4, color='gray', zorder=0)
    ax1.tick_params(axis='y', labelsize=20)

    # Line for SC on twin axis
    ax2 = ax1.twinx()
    ax2.plot(x, sc_vals, color='#333333', lw=4.5, ls='--',
             marker='s', markersize=13, zorder=5)
    for xi, v in zip(x, sc_vals):
        ax2.text(xi + 0.05, v + 0.4, f'{v:.1f}', ha='left', va='bottom',
                 fontsize=22, fontproperties=font_bold, color='#333333')
    ax2.set_ylabel('Safety Compliance (%)', fontproperties=font_bold, fontsize=24)
    ax2.set_ylim(min(sc_vals) * 0.90, max(sc_vals) * 1.12)
    ax2.tick_params(axis='y', labelsize=20)

    # Only 2 legend items
    import matplotlib.patches as mpatch
    from matplotlib.lines import Line2D
    handles = [
        mpatch.Patch(facecolor='#888888', alpha=0.85, label='Success Rate (bar)'),
        Line2D([0],[0], color='#333333', lw=4, ls='--', marker='s',
               markersize=12, label='Safety Compliance (line)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               frameon=False, prop=font_new_roman, fontsize=17,
               bbox_to_anchor=(0.5, 1.01))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_efficiency(df, filename):
    """
    消融实验效率图: Control Cost 柱状图 + Jerk 折线图 (双轴)
    只保留2个图例: Control Cost (bar) 和 Jerk (line)
    """
    if not filename.endswith('.pdf'):
        filename = filename.replace('.png', '.pdf')
    cfgs   = list(df['Config'].unique())
    x      = np.arange(len(cfgs))
    bw     = 0.55
    colors = [_c(c) for c in cfgs]

    fig, ax1 = plt.subplots(figsize=(12, 7))
    cost_vals = [float(df[df['Config']==c]['Cost'].mean()) for c in cfgs]
    jerk_vals = [float(df[df['Config']==c]['Jerk'].mean()) for c in cfgs]

    # Bars for Cost (raw values, not normalized)
    bars = ax1.bar(x, cost_vals, bw, color=colors, alpha=0.85,
                   edgecolor='white', linewidth=1.5, zorder=3)
    for bar, v in zip(bars, cost_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{v:.1f}', ha='center', va='bottom', fontsize=22,
                 fontproperties=font_bold)

    ax1.set_ylabel('Control Cost', fontproperties=font_bold, fontsize=24)
    ax1.set_ylim(0, max(cost_vals) * 1.35)
    ax1.set_xticks(x)
    ax1.set_xticklabels(cfgs, fontproperties=font_new_roman, fontsize=22,
                         rotation=10, ha='right')
    ax1.grid(True, axis='y', ls=':', alpha=0.4, color='gray', zorder=0)
    ax1.tick_params(axis='y', labelsize=20)

    # Line for Jerk on twin axis
    ax2 = ax1.twinx()
    ax2.plot(x, jerk_vals, color='#333333', lw=4.5, ls='--',
             marker='o', markersize=13, zorder=5)
    for xi, v in zip(x, jerk_vals):
        ax2.text(xi + 0.05, v + 1.0, f'{v:.1f}', ha='left', va='bottom',
                 fontsize=22, fontproperties=font_bold, color='#333333')
    ax2.set_ylabel('Jerk', fontproperties=font_bold, fontsize=24)
    ax2.set_ylim(min(jerk_vals) * 0.85, max(jerk_vals) * 1.15)
    ax2.tick_params(axis='y', labelsize=20)

    # Only 2 legend items
    import matplotlib.patches as mpatch
    from matplotlib.lines import Line2D
    handles = [
        mpatch.Patch(facecolor='#888888', alpha=0.85, label='Control Cost (bar)'),
        Line2D([0],[0], color='#333333', lw=4, ls='--', marker='o',
               markersize=12, label='Jerk (line)'),
    ]
    fig.legend(handles=handles, loc='upper center', ncol=2,
               frameon=False, prop=font_new_roman, fontsize=17,
               bbox_to_anchor=(0.5, 1.01))

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), dpi=300, bbox_inches='tight')
    plt.close()


def plot_ablation_summary(df, filename):
    """向后兼容别名: 同时生成reliability和efficiency两张图"""
    base = filename.replace('.pdf', '')
    plot_ablation_reliability(df, base + '_reliability.pdf')
    plot_ablation_efficiency(df,  base + '_efficiency.pdf')
# Alias kept for backward compatibility
plot_ablation_combined = plot_ablation_summary


def _plot_grouped_bars(df, group_col, groups, metrics, filename):
    n_met = len(metrics)
    fig, axs = plt.subplots(1, n_met, figsize=(9*n_met, 8))
    if n_met == 1: axs = [axs]
    bw = 0.7 / max(len(groups), 1)
    xs = np.arange(1)
    for ax, (col_key, ylabel) in zip(axs, metrics):
        for j, g in enumerate(groups):
            sub = df[df[group_col]==g]
            val = float(sub[col_key].mean()) if not sub.empty else 0.0
            offset = (j - len(groups)/2 + 0.5) * bw
            ax.bar(xs+offset, [val], bw*0.88, color=_c(g),
                   label=g, zorder=3, edgecolor='white', linewidth=0.8)
        ax.set_ylabel(ylabel, fontproperties=font_bold, fontsize=26)
        ax.set_xticks([])
        ax.grid(True, axis='y', ls=':', alpha=0.5, color='gray')
        ax.tick_params(labelsize=22)
        ax.legend(frameon=False, prop=font_new_roman, fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), dpi=300)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Privacy–utility trade-off
# ─────────────────────────────────────────────────────────────────────────────
def plot_privacy_tradeoff(df, filename):
    if not filename.endswith('.pdf'):
        filename = filename.replace('.png', '.pdf')

    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    metric_cols = [('Final MSE','Validation MSE'),
                   ('SC (%)','Safety Compliance (%)'),
                   ('SR (%)','Success Rate (%)')]

    eps_vals_raw = df['Target ε'].unique()   # strings like '0.5','1.0','∞'
    # sort: numeric first, then ∞
    def sort_key(v):
        try: return float(v)
        except: return 1e9
    eps_vals = sorted(eps_vals_raw, key=sort_key)
    xs = np.arange(len(eps_vals))

    for ax, (col_key, ylabel) in zip(axes, metric_cols):
        for m in ['DP-FedAvg', 'DP-FedProx']:
            sub  = df[df['Method']==m]
            vals = []
            for ev in eps_vals:
                row = sub[sub['Target ε']==ev]
                vals.append(float(row[col_key].values[0]) if not row.empty else np.nan)
            ax.plot(xs, vals, color=_c(m), lw=4.5,
                    ls=_ls(m), marker=_m(m), markersize=13, label=m)

        ax.set_xticks(xs)
        ax.set_xticklabels([f'ε={v}' for v in eps_vals],
                           fontsize=22, fontproperties=font_new_roman)
        ax.set_ylabel(ylabel, fontproperties=font_bold, fontsize=26)
        ax.set_xlabel('Privacy Budget ε', fontproperties=font_bold, fontsize=26)
        ax.legend(frameon=False, prop=font_new_roman, fontsize=22)
        ax.grid(True, ls=':', alpha=0.5, color='gray')
        ax.tick_params(labelsize=22, width=2, length=6)

    # annotate: ε=∞ = no DP
    for ax in axes:
        ylo, yhi = ax.get_ylim()
        ax.axvline(len(eps_vals)-1, color='#888888', ls='--', lw=2, alpha=0.6)
        ax.text(len(eps_vals)-1.45, ylo + (yhi-ylo)*0.06,
                'No DP', fontsize=18, color='#555555',
                fontproperties=font_new_roman)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename),
                bbox_inches='tight', dpi=300)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Parameter sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def plot_sensitivity(sens, filename):
    """
    参数敏感性图: SR(%) 折线 + SC(%) 折线 (双Y轴)
    SR和SC均为越高越好。
    标注: 在各点旁边显示数值; 标注 "higher is better ↑"
    """
    if not filename.endswith('.pdf'):
        filename = filename.replace('.png', '.pdf')

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    pcolor = _c('FedRMPC');  scolor = _c('Robust MPC')

    for ax, xvals, sr, sc, xlabel in [
        (axes[0], sens['mu_values'],     sens['mu_sr'],  sens['mu_sc'],  'Proximal Coefficient μ'),
        (axes[1], sens['lambda_values'], sens['lam_sr'], sens['lam_sc'], 'Uncertainty Weight λ₀'),
    ]:
        sr_pct = [v * 100 for v in sr]
        ax2 = ax.twinx()
        l1, = ax.plot(xvals, sr_pct, color=pcolor, lw=5, marker='o', ms=14,
                      label='Success Rate (%)')
        l2, = ax2.plot(xvals, sc, color=scolor, lw=5, ls='--',
                       marker='s', ms=14, label='Safety Compliance (%)')
        # value annotations
        for xi, v in zip(xvals, sr_pct):
            ax.text(xi, v + 1.5, f'{v:.1f}%', ha='center', va='bottom',
                    fontsize=18, color=pcolor, fontproperties=font_new_roman)
        for xi, v in zip(xvals, sc):
            ax2.text(xi, v - 1.8, f'{v:.1f}%', ha='center', va='top',
                     fontsize=18, color=scolor, fontproperties=font_new_roman)

        ax.set_xlabel(xlabel, fontproperties=font_bold, fontsize=26)
        ax.set_ylabel('Success Rate (%) ↑', color=pcolor,
                      fontproperties=font_bold, fontsize=24)
        ax2.set_ylabel('Safety Compliance (%) ↑', color=scolor,
                       fontproperties=font_bold, fontsize=24)
        ax.set_ylim(0, max(sr_pct) * 1.30)
        ax2.set_ylim(min(sc) * 0.93, max(sc) * 1.08)
        ax.tick_params(labelsize=22, colors=pcolor)
        ax2.tick_params(labelsize=22, colors=scolor)
        ax.grid(True, ls=':', alpha=0.5, color='gray')
        ax.legend([l1, l2], [l1.get_label(), l2.get_label()],
                  frameon=False, prop=font_new_roman, fontsize=20,
                  loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename),
                bbox_inches='tight', dpi=300)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  Uncertainty heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_uncertainty_surface(model, env, filename='exp1_uncertainty_heatmap.pdf'):
    import torch
    if not filename.endswith('.pdf'):
        filename = filename.replace('.png', '.pdf')

    x = np.linspace(-20, 20, 100)
    y = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(x, y)
    states  = np.zeros((10000, 4), dtype=np.float32)
    states[:,0] = X.flatten();  states[:,1] = Y.flatten();  states[:,2] = 5.0
    actions = np.zeros((10000, 2), dtype=np.float32)

    # New single-arg BNN interface: concatenate (N,6) and call model.forward(xu)
    # Chunked to avoid OOM; each chunk does ONE (M*C, 6) batched forward pass
    CHUNK = 500
    M     = Config.MC_SAMPLES
    all_var = []
    model.train()
    with torch.no_grad():
        for start in range(0, 10000, CHUNK):
            s_c = states[start:start+CHUNK]
            a_c = actions[start:start+CHUNK]
            xu  = torch.FloatTensor(
                np.concatenate([s_c, a_c], axis=1)
            ).to(Config.DEVICE)                          # (C, 6)
            C   = xu.shape[0]
            xu_exp  = xu.unsqueeze(0).expand(M, C, 6).reshape(M * C, 6)
            preds_c = model(xu_exp).view(M, C, 4)        # (M, C, 4)
            var_c   = preds_c.var(dim=0).mean(dim=1)     # (C,)
            all_var.append(var_c.cpu().numpy())
    epistemic = np.concatenate(all_var).reshape(100, 100)
    obstacle_risk = np.zeros_like(epistemic)
    for obs in env.obstacles:
        d = np.sqrt((X - obs['x']) ** 2 + (Y - obs['y']) ** 2) - obs['r']
        obstacle_risk = np.maximum(
            obstacle_risk,
            np.exp(-np.maximum(d, 0.0) / max(Config.SAFETY_MARGIN, 1e-6))
        )
    e_lo, e_hi = np.percentile(epistemic, [5, 99])
    epistemic_n = np.clip((epistemic - e_lo) / max(e_hi - e_lo, 1e-9), 0.0, 1.0)
    Z = np.clip(0.35 * epistemic_n + 0.65 * obstacle_risk, 0.0, 1.0)

    import matplotlib.colors as mcolors
    heat_cmap = mcolors.LinearSegmentedColormap.from_list(
        'white_to_fedred', ['#FFFFFF', '#FFE1E1', '#FF9B9B', '#FF4F4F'])

    fig, ax = plt.subplots(figsize=(15, 13))
    mesh = ax.pcolormesh(X, Y, Z, cmap=heat_cmap, shading='auto',
                         vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel(r'Safety-weighted Uncertainty $\mathcal{U}_i$',
                       fontproperties=font_bold, fontsize=34)
    cbar.ax.tick_params(labelsize=28)

    for obs in env.obstacles:
        ax.add_patch(plt.Circle((obs['x'],obs['y']), obs['r'],
                                edgecolor='black', facecolor='none',
                                lw=2.5, zorder=10))

    ax.set_xlabel('X (m)', fontproperties=font_bold, fontsize=46)
    ax.set_ylabel('Y (m)', fontproperties=font_bold, fontsize=46)
    ax.tick_params(labelsize=38, width=4, length=10)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, filename), dpi=300)
    plt.close()
