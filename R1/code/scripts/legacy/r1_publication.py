"""Build R1 manuscript evidence directly from checkpointed experiment records.

No controller configuration or experimental outcome is changed here. Run after
r1_experiments.py and r1_prediction_horizon.py, or use --partial for finished
panels while the remaining experiments continue.
"""
from __future__ import annotations

from project_paths import R1_ROOT, artifact_path, source_file, source_digest, verify_source_hash, check_protocol

import argparse
import json
import shutil
from pathlib import Path

import r1_experiments as r  # workstation compatibility and plotting defaults
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

ROOT = R1_ROOT
OUT = r.OUT
PAPER = ROOT.parents[1] / 'paper' / 'IoTJ_FedUMPC' / 'R1'
SECTIONS = PAPER / 'sections' / 'generated'
FIGURES = PAPER / 'figures' / 'Experiment' / 'Revision'
DATA = PAPER / 'data' / 'Experiment' / 'Revision'
for directory in [SECTIONS, FIGURES, DATA]:
    directory.mkdir(parents=True, exist_ok=True)

METHODS = ['Linear MPC', 'Tube-style MPC', 'Scenario MPC', 'GP-margin MPC',
           'Fixed-margin MPC', 'Ensemble UMPC', 'Centralized UMPC', 'FedUMPC']
COLORS = dict(zip(METHODS, r.PALETTE + [r.FED_RED]))
SHORT = {'Linear MPC': 'Nominal', 'Tube-style MPC': 'Tube',
         'Scenario MPC': 'Scenario', 'GP-margin MPC': 'GP-margin',
         'Fixed-margin MPC': 'Fixed', 'Ensemble UMPC': 'Ensemble',
         'Centralized UMPC': 'Central', 'FedUMPC': 'FedUMPC'}
DISPLAY = {'Linear MPC': 'Nominal MPC'}
LEARN_COLORS = {'Local Only': '#FFAA53', 'FedAvg': '#50CC55',
                'Prox-FL': '#FF6666', 'Centralized': '#3399FF'}
TABLES = {}
FINDINGS = {}

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8,
                     'xtick.labelsize': 7.2, 'ytick.labelsize': 7.2,
                     'axes.labelpad': 2, 'legend.fontsize': 7.2,
                     'lines.linewidth': 0.85, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'savefig.bbox': None})


def read(name):
    return pd.read_csv(OUT / f'{name}.csv')


def write_tex(name, text):
    (SECTIONS / f'{name}.tex').write_text(text + '\n', encoding='utf-8')


def plain(text):
    return (text.replace(r'\%', '%').replace(r'\,', ' ')
            .replace(r'\mathrm', '').replace('~', ' '))


def finding(name, latex, response=None):
    write_tex(name + '_findings', latex)
    FINDINGS[name.upper()] = response or plain(latex)


def table(name, caption, label, headers, rows, *, wide=False, align=None):
    env = 'table*' if wide else 'table'
    width = r'\textwidth' if wide else r'\columnwidth'
    alignment = align or 'l' + 'c' * (len(headers) - 1)
    lines = [f'\\begin{{{env}}}[!t]', r'\centering',
             '\\caption{\\rev{' + caption + '}}', '\\label{' + label + '}',
             r'\footnotesize', r'\setlength{\tabcolsep}{2.2pt}',
             r'\begin{tabular*}{' + width + '}{@{\\extracolsep{\\fill}}'
             + alignment + '@{}}', r'\toprule',
             ' & '.join(headers) + r'\\', r'\midrule']
    lines += [' & '.join(map(str, row)) + r'\\' for row in rows]
    lines += [r'\bottomrule', r'\end{tabular*}', f'\\end{{{env}}}']
    write_tex(name, '\n'.join(lines))
    TABLES[name] = {'caption': plain(caption), 'label': label,
                    'headers': [plain(x) for x in headers], 'rows': rows}


def msd(row, key, digits=1):
    return f'{row[key+"_mean"]:.{digits}f} ({row[key+"_sd"]:.{digits}f})'


def axes_grid(rows, cols, height):
    fig, axs = plt.subplots(rows, cols, figsize=(7.16, height),
                            layout='constrained')
    fig.set_constrained_layout_pads(w_pad=0.045, h_pad=0.06,
                                    wspace=0.045, hspace=0.08)
    for ax, letter in zip(np.asarray(axs).ravel(), 'abcdefghijkl'):
        ax.text(-0.13, 1.055, f'({letter})', transform=ax.transAxes,
                va='bottom', fontsize=8.3, fontweight='bold')
        ax.tick_params(pad=2, length=2.5)
        ax.grid(axis='y', alpha=0.25, ls=':', lw=0.5)
        ax.set_axisbelow(True)
    return fig, axs


def save(fig, name):
    fig.savefig(FIGURES / f'{name}.pdf')
    fig.savefig(OUT / f'{name}.pdf')
    plt.close(fig)


def comparison():
    common = read('r1_main_summary').set_index('Method')
    mismatch = read('r1_mismatch_summary').set_index('Method')
    rows = []
    for method in METHODS[:-1] + ['Local UMPC', 'FedAvg UMPC', 'FedUMPC']:
        row = [DISPLAY.get(method, method)]
        for frame in [common, mismatch]:
            row += ([msd(frame.loc[method], m) for m in ['SR', 'SC', 'CR']]
                    if method in frame.index else ['---'] * 3)
        rows.append(row)
    table('comparison_tables',
          'Control Endpoints on Ten Paired Scenes per Plant Regime. '
          'Values Are Mean (Sample SD) in Percent; Each Method Has 40 Routes. '
          'SR Is Completion, SC Is Clearance Compliance, and CR Is Route Contact Rate.',
          'tab:r1_comparison',
          ['Method', r'Common SR $\uparrow$', r'Common SC $\uparrow$',
           r'Common CR $\downarrow$', r'Mismatch SR $\uparrow$',
           r'Mismatch SC $\uparrow$', r'Mismatch CR $\downarrow$'], rows,
          wide=True)
    f = common.loc['FedUMPC']; fixed = common.loc['Fixed-margin MPC']
    contrast = read('r1_main_paired_contrasts').set_index('Comparator')
    d = contrast.loc['Fixed-margin MPC']
    matched = mismatch.loc[['Local UMPC', 'FedAvg UMPC', 'FedUMPC']]
    text = (f'Table~\\ref{{tab:r1_comparison}} does not establish a general advantage '
            f'of the full controller. On the common plant, FedUMPC records '
            f'{f.SR_mean:.1f}\\% SR and {f.SC_mean:.1f}\\% SC, compared with '
            f'{fixed.SR_mean:.1f}\\% and {fixed.SC_mean:.1f}\\% for the non-learned '
            f'fixed-margin reference. The paired FedUMPC-minus-fixed differences '
            f'are {d.Delta_SR:.1f} percentage points (pp), 95\\% CI '
            f'$[{d.Delta_SR_ci_lo:.1f},{d.Delta_SR_ci_hi:.1f}]$, for SR and '
            f'{d.Delta_SC:.2f}~pp, CI '
            f'$[{d.Delta_SC_ci_lo:.2f},{d.Delta_SC_ci_hi:.2f}]$, for SC. '
            'The fixed-margin reference therefore exceeds the full controller '
            'on both primary endpoints in this evaluated regime. The poor '
            'scenario-MPC outcome is specific to the approximate instantiation '
            'and cannot characterize canonical stochastic MPC. Under strong mismatch, Local, '
            f'FedAvg, and Prox-FL UMPC attain SRs of '
            f'{matched.SR_mean.iloc[0]:.1f}/{matched.SR_mean.iloc[1]:.1f}/'
            f'{matched.SR_mean.iloc[2]:.1f}\\% and SCs of '
            f'{matched.SC_mean.iloc[0]:.2f}/{matched.SC_mean.iloc[1]:.2f}/'
            f'{matched.SC_mean.iloc[2]:.2f}\\%. Federation therefore has no '
            'demonstrated independent closed-loop advantage in this matched test.')
    finding('comparison', text,
            f'On ten paired common-plant scenes, FedUMPC achieved {f.SR_mean:.1f}% '
            f'SR/{f.SC_mean:.2f}% SC, while fixed-margin MPC achieved '
            f'{fixed.SR_mean:.1f}%/{fixed.SC_mean:.2f}%. Paired differences were '
            f'{d.Delta_SR:.1f} pp (95% CI [{d.Delta_SR_ci_lo:.1f}, '
            f'{d.Delta_SR_ci_hi:.1f}]) and {d.Delta_SC:.2f} pp '
            f'([{d.Delta_SC_ci_lo:.2f}, {d.Delta_SC_ci_hi:.2f}]). '
            'We withdrew the earlier superiority interpretation.')
    FINDINGS['MATCHED'] = (f'With the same uncertainty-aware controller on the '
        'mismatched plants, Local/FedAvg/Prox-FL yielded '
        + '/'.join(f'{v:.1f}' for v in matched.SR_mean) + '% SR and '
        + '/'.join(f'{v:.2f}' for v in matched.SC_mean)
        + '% SC; these results do not establish an independent Prox-FL control gain.')

    fig, axs = axes_grid(2, 2, 4.7)
    a, b, c, e = axs.ravel()
    cmp = contrast.reindex(METHODS[:-1]).iloc[::-1]
    y = np.arange(len(cmp))
    a.errorbar(cmp.Delta_SC, y,
                xerr=[cmp.Delta_SC - cmp.Delta_SC_ci_lo,
                      cmp.Delta_SC_ci_hi - cmp.Delta_SC], fmt='o',
                color=r.FED_RED, ecolor='#4D4D4D', capsize=2, lw=0.8, ms=3.7)
    a.axvline(0, lw=0.7, ls='--', color='#555555')
    a.set_yticks(y, [SHORT[x] for x in cmp.index]); a.set_xlabel('SC difference (pp)\nFedUMPC minus comparator')
    a.grid(axis='x', alpha=0.25, ls=':')
    routes = read('r1_main_routes')
    for method in ['Linear MPC', 'Fixed-margin MPC', 'Centralized UMPC', 'FedUMPC']:
        v = np.sort(routes.loc[routes.Method == method, 'min_clearance'])
        b.step(v, np.arange(1, len(v)+1)/len(v), where='post',
               label=SHORT[method], color=COLORS[method])
    b.axvline(1.35, lw=0.7, ls='--', color='#555555')
    b.set_xlabel('Route minimum clearance (m)'); b.set_ylabel('ECDF (fraction)')
    b.legend(frameon=False, ncol=2, loc='lower right', handlelength=1.3)
    x = np.arange(len(METHODS))
    c.bar(x, common.reindex(METHODS).MeanViolationDepth_mean,
          color=[COLORS[m] for m in METHODS], width=0.65, alpha=0.8)
    c2 = c.twinx()
    c2.plot(x, 0.1 * common.reindex(METHODS).MaxViolationDuration_mean,
            'o--', color='#4D4D4D', ms=3)
    c.set_xticks(x, [SHORT[m] for m in METHODS], rotation=65, ha='right')
    c.set_ylabel('Violation depth (m)'); c2.set_ylabel('Longest episode (s)', labelpad=2)
    c2.tick_params(labelsize=7, pad=2)
    horizon = read('r1_prediction_horizon')
    rng = np.random.default_rng(20261107)
    colors = ['#FFAA53', '#FF6666']
    for k, (column, title) in enumerate([
            ('Physics_position_error_m', 'Nominal physics'),
            ('Hybrid_position_error_m', 'Hybrid')]):
        matrix = horizon.pivot(index='Query', columns='Time_s', values=column)
        arr = matrix.to_numpy(); draws = rng.integers(0, len(arr), (2000, len(arr)))
        boots = arr[draws].mean(axis=1)
        low, high = np.percentile(boots, [2.5, 97.5], axis=0)
        e.plot(matrix.columns, arr.mean(axis=0), color=colors[k], label=title)
        e.fill_between(matrix.columns, low, high, color=colors[k], alpha=0.18)
    e.set_xlabel('Prediction horizon (s)'); e.set_ylabel('Position error (m)')
    e.legend(frameon=False, loc='upper left')
    save(fig, 'r1_control_diagnostics')
    last = horizon[horizon.Horizon_step == horizon.Horizon_step.max()]
    finding('horizon',
        f'At $2.5$~s, mean position errors are '
        f'${last.Physics_position_error_m.mean():.3f}$~m for nominal physics and '
        f'${last.Hybrid_position_error_m.mean():.3f}$~m for the hybrid, so the '
        'learned correction does not improve horizon prediction in this test.',
        'In 128 independent held-out candidate rollouts, the 2.5 s mean position '
        f'error was {last.Physics_position_error_m.mean():.3f} m for physics and '
        f'{last.Hybrid_position_error_m.mean():.3f} m for the hybrid. '
        'The interior disagreement inequality held on 3177 query horizons, '
        'but the hybrid was less accurate; no beneficial error bound is claimed.')


def mechanisms():
    frame = read('r1_mechanisms_summary').set_index('Method')
    paired = read('r1_mechanisms_paired_contrasts').set_index('Comparator')
    order = ['No uncertainty response', 'Risk only', 'Trust only', 'Margin only',
             'Risk + trust', 'Risk + margin', 'Trust + margin', 'Constant U',
             'Shuffled U', 'Fixed-growth MPC', 'Full FedUMPC']
    rows = []
    for method in order:
        row = frame.loc[method]
        rows.append([method, msd(row, 'SR'), msd(row, 'SC'), msd(row, 'CR')])
    table('mechanism_table', 'Mechanism Isolation on Ten Paired Common-Plant '
          'Scenes. Values Are Mean (Sample SD) in Percent.', 'tab:r1_mechanisms',
          ['Configuration', r'SR $\uparrow$', r'SC $\uparrow$', r'CR $\downarrow$'], rows)
    full = frame.loc['Full FedUMPC']; fixed = frame.loc['Fixed-growth MPC']
    contrast=paired.loc['Fixed-growth MPC']
    text = ('Table~\\ref{tab:r1_mechanisms} separates response magnitude from '
            'candidate-specific uncertainty ordering. ' +
            '; '.join(f'{m} gives {frame.loc[m,"SR_mean"]:.1f}\\% SR/'
                      f'{frame.loc[m,"SC_mean"]:.2f}\\% SC'
                      for m in ['Full FedUMPC', 'Constant U', 'Shuffled U'])
            + '. Removing the direct risk term (trust plus margin) leaves all '
            'reported control endpoints identical to the full controller, '
            'and risk plus margin matches margin only. The direct risk term '
            'therefore has no demonstrated incremental effect in these tests.'
            + f' The non-learned matched-growth profile gives '
            f'{fixed.SR_mean:.1f}\\% SR/{fixed.SC_mean:.2f}\\% SC. '
            f'The full-minus-fixed-growth SC difference is '
            f'${contrast.Delta_SC:.2f}$~pp, 95\\% CI '
            f'$[{contrast.Delta_SC_ci_lo:.2f},{contrast.Delta_SC_ci_hi:.2f}]$, '
            'whereas the SR difference is $-10.0$~pp, CI $[-17.5,-2.5]$. '
            'The three-part combination therefore has no demonstrated '
            'incremental benefit over a mean-matched deterministic profile. '
            'Finite-sample dropout variability and the retained model-discrepancy '
            'penalty also distinguish these interventions from a calibrated '
            'uncertainty-to-risk experiment.')
    finding('mechanism', text,
            'The full controller attained '
            f'{full.SR_mean:.1f}% SR/{full.SC_mean:.2f}% SC. '
            + '; '.join(f'{m}: {frame.loc[m,"SR_mean"]:.1f}%/'
                        f'{frame.loc[m,"SC_mean"]:.2f}%'
                        for m in ['Risk only', 'Trust only', 'Margin only',
                                  'Risk + trust', 'Risk + margin', 'Trust + margin',
                                  'Constant U', 'Shuffled U', 'Fixed-growth MPC'])
            + '. Removing direct risk leaves the reported endpoints identical '
            'to the full method. These results replace the former all-at-once '
            'ablation and do not establish the necessity of the full combination.')


def uncertainty():
    frame = read('r1_uncertainty_samples')
    summary = read('r1_uncertainty_summary').set_index('Scenario')
    pooled = summary.loc['Pooled']
    q_u = frame.Uncertainty.quantile(.25); q_e = frame.Prediction_error.quantile(.75)
    false_low = (frame.Uncertainty <= q_u) & (frame.Prediction_error >= q_e)
    conditional = 100 * false_low.sum() / (frame.Uncertainty <= q_u).sum()
    finding('uncertainty',
        f'The pooled Spearman correlation is ${pooled.Spearman_rho:.3f}$, '
        f'with scenario values spanning ${summary.iloc[:4].Spearman_rho.min():.3f}$--'
        f'${summary.iloc[:4].Spearman_rho.max():.3f}$. False-low events occupy '
        f'${100*false_low.mean():.3f}\\%$ of all queries but '
        f'${conditional:.1f}\\%$ of the low-uncertainty quartile. Their joint '
        'frequency is $2.2/2.7/4.0/4.8\\%$ in nominal/noisy/mass-shift/wind '
        'conditions. Mean $\\mathcal U$ changes little across these conditions '
        '($0.130$--$0.133$), although prediction errors differ. The signal is '
        'therefore a weak-to-moderate error-ranking cue, neither a calibrated '
        'error radius nor a validated shift detector.',
        f'Across 8000 held-out queries, pooled Spearman rho was '
        f'{pooled.Spearman_rho:.3f}. False-low events occurred in '
        f'{100*false_low.mean():.3f}% of all queries and {conditional:.1f}% '
        'of queries in the lowest-U quartile. Mean U remained 0.130–0.133 '
        'across shifts; uncertainty is not a calibrated error or shift bound.')
    # Four panels are constructed even before the MC sweep finishes, with no
    # final file saved until both MC diagnostics exist.
    if not (OUT / 'r1_mc_closed_loop_summary.csv').exists():
        return
    mc = read('r1_mc_closed_loop_summary'); query = read('r1_mc_query_diagnostics')
    fig, axs = axes_grid(2, 3, 4.95)
    a,b,c,d,e,f = axs.ravel()
    a.grid(False)
    histogram, xe, ye = np.histogram2d(frame.Uncertainty, frame.Prediction_error,
                                      bins=(35,35))
    image = a.pcolormesh(xe, ye, histogram.T, cmap=r.POSITIVE, vmin=0,
                         shading='auto', rasterized=True)
    a.axvline(q_u, color='#007FFF', lw=.7, ls='--')
    a.axhline(q_e, color='#4D4D4D', lw=.7, ls=':')
    a.set_xlabel('Uncertainty U (dimensionless)'); a.set_ylabel('Normalized error (–)')
    a.text(.98,.98, f'$\\rho_s$={pooled.Spearman_rho:.3f}',
            transform=a.transAxes, ha='right', va='top', fontsize=7.3)
    cb=fig.colorbar(image, ax=a, fraction=.05, pad=.02); cb.set_label('Count', size=7)
    bins = pd.qcut(frame.Uncertainty,4,labels=['Q1','Q2','Q3','Q4'])
    values = [frame.loc[bins==q,'Prediction_error'].to_numpy() for q in ['Q1','Q2','Q3','Q4']]
    violin=b.violinplot(values, showextrema=False, showmedians=True)
    for body,color in zip(violin['bodies'],r.PALETTE[:3]+[r.FED_RED]):
        body.set_facecolor(color);body.set_alpha(.65)
    violin['cmedians'].set_color('#4D4D4D');violin['cmedians'].set_linewidth(.7)
    b.set_xticks([1,2,3,4],['Q1','Q2','Q3','Q4']);b.set_xlabel('Uncertainty quartile')
    b.set_ylabel('Normalized error (–)')
    for v,q,color in zip(values,['Q1','Q2','Q3','Q4'],r.PALETTE[:3]+[r.FED_RED]):
        levels=np.linspace(0,1,100)
        c.plot(levels, np.quantile(v,levels), color=color,label=q)
    c.set_xlabel('Error quantile (fraction)');c.set_ylabel('Normalized error (–)')
    c.legend(frameon=False,ncol=2, handlelength=1.2,loc='upper left')
    names=['Nominal','High noise','Mass shift','Wind']
    x=np.arange(4); s=summary.loc[names]
    d.bar(x,s.False_low_uncertainty_pct,color=r.PALETTE[:3]+[r.FED_RED],alpha=.8)
    d2=d.twinx();d2.plot(x,s.Spearman_rho,'o--',color='#4D4D4D',ms=3)
    d.set_xticks(x,['Nominal','High\nnoise','Mass\nshift','Wind'])
    d.set_ylabel('False-low joint frequency (%)');d2.set_ylabel('Spearman correlation (–)',labelpad=2)
    d2.tick_params(labelsize=7,pad=2);d2.set_ylim(0,.4)
    e.plot(query.M,query.Mean_query_repeat_variance,'o-',color=r.FED_RED,ms=3)
    e2=e.twinx();e2.plot(query.M,query.Mean_batch_latency_ms,'s--',color='#3399FF',ms=3)
    e.set_xlabel('MC samples M (count)');e.set_ylabel('Repeat variance of U (–)')
    e.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    e2.set_ylabel('256-query batch time (ms)',labelpad=2);e2.tick_params(labelsize=7,pad=2)
    mc_raw=read('r1_mc_closed_loop_runs')
    for index,(_,row) in enumerate(mc.iterrows()):
        scene_values=mc_raw.loc[mc_raw.M==row.M,'SC'].to_numpy()
        lo,hi=r.cluster_bootstrap(scene_values)
        f.errorbar(row.Mean_controller_ms,row.SC_pct,
                   yerr=[[row.SC_pct-lo],[hi-row.SC_pct]],fmt='none',
                   color=r.FED_RED,lw=.65,capsize=2,alpha=.65)
        f.scatter(row.Mean_controller_ms,row.SC_pct,color=r.FED_RED,s=20+row.M)
        offsets=[(4,-16),(4,18),(-3,-18),(-3,16)]
        f.annotate(f'{int(row.M)} / {row.SR_pct:.0f}%',
                   (row.Mean_controller_ms,row.SC_pct),xytext=offsets[index],
                   ha='left' if index<2 else 'right',
                   textcoords='offset points', fontsize=6.6,
                   arrowprops=dict(arrowstyle='-',lw=.45,color='#4D4D4D'))
    f.margins(x=.28,y=.15);f.set_xlabel('Mean controller call (ms)');f.set_ylabel('Safety compliance (%)')
    save(fig,'r1_uncertainty_diagnostics')
    mcr = read('r1_mc_closed_loop_runs')
    contrasts=[]
    ref=mcr[mcr.M==10].set_index('Seed')
    for m,sub in mcr.groupby('M'):
        sub=sub.set_index('Seed').loc[ref.index]
        result={'M':int(m),'Reference_M':10,'Paired_scenes':len(ref)}
        for metric,factor in [('SR',100),('SC',1),('ComputeMs',1)]:
            delta=factor*(sub[metric]-ref[metric])
            lo,hi=r.cluster_bootstrap(delta.to_numpy())
            result[metric+'_delta']=delta.mean()
            result[metric+'_ci_lo']=lo;result[metric+'_ci_hi']=hi
        contrasts.append(result)
    pd.DataFrame(contrasts).to_csv(OUT/'r1_mc_paired_contrasts.csv',index=False)
    validation=read('r1_mc_validation_summary').set_index('M')
    rows=[]
    for _,row in mc.iterrows():
        q=query[query.M==row.M].iloc[0]
        rows.append([str(int(row.M)),f'{row.SR_pct:.1f}',f'{row.SC_pct:.2f}',
                     f'{row.Mean_controller_ms:.1f}',f'{row.Mean_run_p9_ms:.1f}',
                     f'{row.Mean_run_p90_ms:.1f}',f'{row.Mean_run_p95_ms:.1f}',
                     f'{q.Mean_query_repeat_variance:.2e}',
                     f'{validation.loc[row.M,"Spearman_rho"]:.3f}'])
    table('mc_table','MC Sampling Study on Five Paired Scenes. Latency Columns '
          'Are Means of Per-Run Percentiles, Not Percentiles of Run Means. '
          'Repeat Variance Uses 20 Repetitions of 256 Fixed Queries.',
          'tab:r1_mc',['$M$','SR (\\%)','SC (\\%)','Mean (ms)','P9 (ms)',
          'P90 (ms)','P95 (ms)','Var($U$)','Held-out $\\rho_s$'],rows,wide=True)
    low=query.iloc[0];high=query.iloc[-1]
    pairs=mcr.pivot(index='Seed',columns='M',values='SC')
    delta=pairs[40]-pairs[10];lo,hi=r.cluster_bootstrap(delta)
    finding('mc',
        f'Increasing $M$ from 5 to 40 changes the repeated-query variance '
        f'from ${low.Mean_query_repeat_variance:.2e}$ to '
        f'${high.Mean_query_repeat_variance:.2e}$, while '
        f'the paired SC change from $M=10$ to $M=40$ is '
        f'${delta.mean():.2f}$~pp, 95\\% CI $[{lo:.2f},{hi:.2f}]$. '
        'Table~\\ref{tab:r1_mc} reports both the literal ninth percentile '
        'requested in review and the conventional 90th/95th percentiles; '
        'larger $M$ is not interpreted as automatic control improvement.',
        f'For M=5/10/20/40, SR was '+ '/'.join(f'{v:.1f}' for v in mc.SR_pct)
        +'% and SC was '+ '/'.join(f'{v:.2f}' for v in mc.SC_pct)
        +'%. Mean call latency was '+ '/'.join(f'{v:.1f}' for v in mc.Mean_controller_ms)
        +' ms. Repeated-query variance fell from '
        f'{low.Mean_query_repeat_variance:.2e} to {high.Mean_query_repeat_variance:.2e}; '
        f'the M=40 minus M=10 paired SC CI was [{lo:.2f}, {hi:.2f}] pp. '
        'Literal P9 and conventional P90/P95 timings are all included.')


def federation():
    frame=read('r1_fleet_scaling_summary')
    indexed=frame.set_index(['N','Method'])
    benefit=[]
    for n in [16,32]:
        local=indexed.loc[(n,'Local Only'),'Sparse_client_MSE']
        prox=indexed.loc[(n,'Prox-FL'),'Sparse_client_MSE']
        benefit.append(100*(1-prox/local))
    finding('fleet',
        'The data-poor-client MSE of Prox-FL is $0.0381/0.0310/0.0305$ for '
        '$N=4/16/32$, versus $0.0334/0.0369/0.0394$ for local learning. '
        f'Thus, proximal sharing improves this subgroup by {benefit[0]:.1f}\\% '
        f'and {benefit[1]:.1f}\\% at $N=16$ and 32, but worsens it at $N=4$. '
        'Matched FedAvg is more accurate at every fleet size '
        '($0.0222/0.0140/0.0137$). Prox-FL reduces drift '
        '($0.674/0.686/0.681$ versus FedAvg\'s $1.530/1.503/1.478$) '
        'without reducing fleet-wide error, demonstrating that drift '
        'suppression can accompany harmful regularization bias. Its final '
        '$\\widehat B=1.82/2.10/2.35$ likewise does not verify uniform '
        'dissimilarity or positive descent conditions.',
        f'With N=16/32, Prox-FL reduced data-poor-client normalized MSE over '
        f'local learning by {benefit[0]:.1f}%/{benefit[1]:.1f}%, but it was worse '
        'at N=4. Matched FedAvg was more accurate at every fleet size. '
        'Prox-FL drift was 0.674/0.686/0.681, versus 1.530/1.503/1.478 for '
        'FedAvg; smaller drift did not imply better prediction. Final empirical '
        'B estimates were 1.82/2.10/2.35, not uniform theorem constants.')
    FINDINGS['DRIFT']='Prox-FL drift was 0.674/0.686/0.681 for N=4/16/32, compared with FedAvg 1.530/1.503/1.478, despite higher prediction MSE at every fleet size.'
    stress=read('r1_communication_stress_summary').set_index('Scenario')
    full=stress.loc['Synchronous full'];half=stress.loc['50% participation']
    stale=stress.loc['50% + two-round stale']
    finding('communication',
        f'Half participation reduces parameter traffic from '
        f'${full.Round_traffic_MB:.3f}$ to ${half.Round_traffic_MB:.3f}$~MB per '
        f'round, with fleet normalized MSE changing from ${full.Mean_MSE:.4f}$ '
        f'to ${half.Mean_MSE:.4f}$. Adding two-round staleness increases '
        f'error to ${stale.Mean_MSE:.4f}$ '
        f'({100*(stale.Mean_MSE/full.Mean_MSE-1):.1f}\\% above full participation). '
        'This controlled degradation is more informative than payload arithmetic '
        'alone, but it remains a learning-layer stress test rather than '
        'a measured wireless network.',
        f'At N=16, full/half/half-plus-stale participation gave fleet MSE '
        f'{full.Mean_MSE:.4f}/{half.Mean_MSE:.4f}/{stale.Mean_MSE:.4f}, with '
        f'{full.Round_traffic_MB:.3f}/{half.Round_traffic_MB:.3f}/'
        f'{stale.Round_traffic_MB:.3f} MB per round. Staleness worsened error '
        f'by {100*(stale.Mean_MSE/full.Mean_MSE-1):.1f}% over full participation.')
    fig,axs=axes_grid(2,2,4.35);a,b,c,d=axs.ravel()
    for method,color in LEARN_COLORS.items():
        sub=frame[frame.Method==method]
        a.errorbar(sub.N,sub.Sparse_client_MSE,yerr=sub.Sparse_client_MSE_sd,
                    fmt='o-',ms=3,color=color,capsize=2,label=method)
    a.set_xticks([4,16,32]);a.set_xlabel('Fleet size N (clients)')
    a.set_ylabel('Data-poor normalized MSE (–)');a.legend(ncol=2,frameon=False,fontsize=7)
    for method,marker in [('FedAvg','o'),('Prox-FL','s')]:
        sub=frame[frame.Method==method]
        b.scatter(sub.Final_drift,sub.Mean_MSE,color=LEARN_COLORS[method],marker=marker,s=22)
        for position,(_,row) in enumerate(sub.iterrows()):
            offsets = ([(5,8),(-8,5),(-4,-15)] if method=='FedAvg'
                       else [(5,8),(18,0),(5,-15)])
            b.annotate(f'N={row.N}',(row.Final_drift,row.Mean_MSE),
                        xytext=offsets[position],textcoords='offset points',
                        ha='right' if method=='FedAvg' and position>0 else 'left',
                        fontsize=6.8,arrowprops=dict(arrowstyle='-',lw=.5,
                                                   color=LEARN_COLORS[method]))
    b.margins(.20);b.set_xlabel('Broadcast-relative client drift (–)')
    b.set_ylabel('Fleet normalized MSE (–)')
    for method in ['FedAvg','Prox-FL','Centralized']:
        sub=frame[frame.Method==method]
        c.plot(sub.N,sub.Empirical_B,'o-',ms=3,color=LEARN_COLORS[method],label=method)
    c.set_xticks([4,16,32]);c.set_xlabel('Fleet size N (clients)')
    c.set_ylabel('Empirical dissimilarity B (–)');c.legend(frameon=False,ncol=2,fontsize=7)
    names=['Synchronous full','50% participation','50% + two-round stale']
    sub=stress.loc[names];x=np.arange(3)
    d.errorbar(x,sub.Mean_MSE,yerr=sub.Mean_MSE_sd,fmt='o-',color=r.FED_RED,capsize=2)
    d2=d.twinx();d2.plot(x,sub.Round_traffic_MB,'s--',color='#4D4D4D',ms=3)
    d.set_xticks(x,['Full','Half','Half + stale']);d.set_ylabel('Fleet normalized MSE (–)')
    d2.set_ylabel('Parameter traffic (MB/round)',labelpad=2);d2.tick_params(labelsize=7,pad=2)
    save(fig,'r1_federation_diagnostics')


def deployment():
    raw=read('r1_obstacle_sensitivity')
    summary=read('r1_obstacle_sensitivity_summary')
    finding('sensitivity',
        f'Across these sweeps, mean SR spans '
        f'${100*summary.SR.min():.1f}$--${100*summary.SR.max():.1f}\\%$ and '
        f'SC spans ${summary.SC.min():.2f}$--${summary.SC.max():.2f}\\%$. '
        'The observed variation is reported rather than labelled parameter '
        'invariance; three scenes cannot establish robustness outside the '
        'tested ranges or identify a universally preferable potential.',
        f'The one-at-a-time obstacle sweeps cover q_obs=5460/7800/10140, '
        'c_s=0.01/0.02/0.04, inner coefficient=0.80/1.15/1.50, and influence '
        f'floor=3.8/4.2/4.8 m. Mean SR ranges {100*summary.SR.min():.1f}–'
        f'{100*summary.SR.max():.1f}% and SC {summary.SC.min():.2f}–'
        f'{summary.SC.max():.2f}% across three development scenes.')
    timing=read('r1_candidate_latency');main=read('r1_main_summary').set_index('Method')
    fed=main.loc['FedUMPC']
    finding('timing',
        f'FedUMPC\'s mean common-plant controller call is '
        f'${fed.ComputeMs_mean:.1f}$~ms and its mean per-run 95th percentile is '
        f'${fed.ComputeP95Ms_mean:.1f}$~ms. Score-evaluation means over '
        f'$M=5$--40 and 4--16 candidates range from '
        f'${timing.Mean_ms.min():.1f}$ to ${timing.Mean_ms.max():.1f}$~ms, '
        f'with a maximum measured 95th percentile of ${timing.P95_ms.max():.1f}$~ms.',
        f'On the RTX 4080 workstation, mean FedUMPC call time was '
        f'{fed.ComputeMs_mean:.1f} ms and mean per-run P95 was '
        f'{fed.ComputeP95Ms_mean:.1f} ms. Warmed score evaluations ranged '
        f'{timing.Mean_ms.min():.1f}–{timing.Mean_ms.max():.1f} ms over '
        'M=5/10/20/40 and C=4/8/12/16. Sensing, network scheduling, '
        'actuation and embedded timing remain unmeasured.')
    fig,axs=axes_grid(2,3,4.75)
    for ax,parameter,label in zip(axs.ravel()[:4],
            ['q_obs','c_s','inner_a','d_inf'],
            ['$q_{obs}$ (cost/m$^2$)','$c_s$ (–)','$c_{in}$ (–)',
             'Influence floor (m)']):
        sub=raw[raw.Parameter==parameter]
        g=sub.groupby('Value').agg(SR=('SR','mean'),SRsd=('SR','std'),
                                    SC=('SC','mean'),SCsd=('SC','std'))
        ax.errorbar(g.index,100*g.SR,yerr=100*g.SRsd,fmt='o-',capsize=2,
                    ms=3,color=r.FED_RED,label='SR')
        ax.errorbar(g.index,g.SC,yerr=g.SCsd,fmt='s--',capsize=2,
                    ms=3,color='#3399FF',label='SC')
        ax.set_xlabel(label);ax.set_ylabel('Endpoint (%)');ax.margins(x=.12)
    axs[0,0].legend(frameon=False,ncol=2,loc='lower right',handlelength=1.2)
    ax=axs[1,1]
    for (m,sub),color in zip(timing.groupby('M'),r.PALETTE[:3]+[r.FED_RED]):
        ax.plot(sub.Candidates,sub.Mean_ms,'o-',ms=3,color=color,label=f'M={m}')
    ax.axhline(100,color='#4D4D4D',ls='--',lw=.65)
    ax.set_xlabel('Candidates C (count)');ax.set_ylabel('Mean score time (ms)')
    ax.legend(frameon=False,ncol=2,fontsize=6.8,handlelength=1)
    ax=axs[1,2];mat=timing.pivot(index='M',columns='Candidates',values='P95_ms')
    im=ax.imshow(mat,cmap=r.POSITIVE,vmin=0,aspect='auto')
    ax.set_xticks(range(4),mat.columns);ax.set_yticks(range(4),mat.index)
    ax.set_xlabel('Candidates C (count)');ax.set_ylabel('MC samples M (count)');ax.grid(False)
    for (i,j),val in np.ndenumerate(mat.to_numpy()):
        ax.text(j,i,f'{val:.0f}',ha='center',va='center',fontsize=7)
    cb=fig.colorbar(im,ax=ax,fraction=.05,pad=.02);cb.set_label('P95 (ms)',size=7)
    save(fig,'r1_deployment_diagnostics')


def mujoco():
    source=ROOT.parents[1]/'MuJoCo'/'R1'/'outputs'
    frame=pd.read_csv(source/'r1_mujoco_multiseed_summary.csv')
    rows=[]
    mapping={'Linear MPC':'Nominal MPC','Tube MPC':'Tube-style MPC',
             'Stochastic MPC':'Scenario MPC','GP-MPC':'GP-margin MPC',
             'Robust MPC':'Fixed-margin MPC'}
    for _,row in frame.iterrows():
        rows.append([mapping.get(row.method,row.method),
            f'{25*row.success_mean:.1f}',f'{row.violation_frequency_pct_mean:.2f}',
            f'{row.mean_violation_depth_mean:.3f}',
            f'{row.max_violation_duration_s_mean:.2f}'])
    table('mujoco_table','MuJoCo Means over Five Independent Scenes '
          '(20 Routes per Method). V Is Violation Frequency, D Is Conditional '
          'Mean Depth, and T Is Maximum Consecutive Violation Duration.',
          'tab:r1_mujoco',['Method','SR (\\%)','V (\\%)','D (m)','T (s)'],rows)
    fed=frame[frame.method=='FedUMPC'].iloc[0]
    nominal=frame[frame.method=='Linear MPC'].iloc[0]
    raw=pd.read_csv(source/'r1_mujoco_multiseed.csv')
    success=raw.pivot(index='seed',columns='method',values='success')
    violation=raw.pivot(index='seed',columns='method',values='violation_frequency_pct')
    delta_success=25*(success.FedUMPC-success['Linear MPC'])
    delta_violation=violation.FedUMPC-violation['Linear MPC']
    srlo,srhi=r.cluster_bootstrap(delta_success)
    vlo,vhi=r.cluster_bootstrap(delta_violation)
    finding('mujoco',
        f'Across five scenes, FedUMPC completes {25*fed.success_mean:.1f}\\% '
        f'of routes, compared with {25*nominal.success_mean:.1f}\\% for '
        f'nominal MPC, and has {fed.collisions_mean:.1f} contact routes per '
        f'scene on average. Its violation frequency is the lowest observed '
        f'(${fed.violation_frequency_pct_mean:.2f}\\%$), with '
        f'${fed.mean_violation_depth_mean:.3f}$~m conditional mean depth and '
        f'${fed.max_violation_duration_s_mean:.2f}$~s maximum-duration mean. '
        f'The paired FedUMPC-minus-nominal differences are '
        f'${delta_success.mean():.1f}$~pp in SR, 95\\% CI '
        f'$[{srlo:.1f},{srhi:.1f}]$, and ${delta_violation.mean():.2f}$~pp '
        f'in violation frequency, CI $[{vlo:.2f},{vhi:.2f}]$. '
        'These small-sample intervals do not establish a general advantage. '
        'In the displayed seed-123 scene, '
        'FedUMPC completes three routes and records one contact, so the '
        'qualitative panel is not selected as a success-only example.',
        f'On five MuJoCo scenes, FedUMPC achieved {25*fed.success_mean:.1f}% '
        f'completion versus nominal MPC {25*nominal.success_mean:.1f}%, '
        f'{fed.collisions_mean:.1f} contact routes per scene, '
        f'{fed.violation_frequency_pct_mean:.2f}% threshold-violation frequency, '
        f'{fed.mean_violation_depth_mean:.3f} m conditional depth and '
        f'{fed.max_violation_duration_s_mean:.2f} s mean longest episode. '
        f'Paired FedUMPC-minus-nominal SR/violation-frequency CIs were '
        f'[{srlo:.1f}, {srhi:.1f}]/[{vlo:.2f}, {vhi:.2f}] pp. '
        'The displayed seed 123 has three completed routes and one contact; '
        'we removed the earlier contact-free interpretation.')
    for path in source.glob('r1_mujoco*.csv'):
        shutil.copy2(path,DATA/path.name)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    comparison();uncertainty();federation();mujoco()
    for name,fn in [('r1_mechanisms_summary.csv',mechanisms),
                    ('r1_obstacle_sensitivity_summary.csv',deployment)]:
        if (OUT/name).exists(): fn()
        elif not args.partial: raise FileNotFoundError(name)
    for path in OUT.glob('r1_*.csv'):
        shutil.copy2(path,DATA/path.name)
    shutil.copy2(OUT/'r1_metadata.json',DATA/'r1_metadata.json')
    (DATA/'response_evidence.json').write_text(json.dumps(
        {'tables':TABLES,'findings':FINDINGS},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Publication evidence generated:',len(TABLES),'tables,',len(FINDINGS),'findings')


if __name__=='__main__':
    main()
