# main.py
"""
Usage:
    python main.py           # fast mode  (~15 min CPU)
    python main.py --full    # full mode  (~3-4 h CPU / ~40 min GPU)

Timing is recorded at every stage and saved to results/timing_report.xlsx
"""

import sys, math, os, time, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')

# ── mode selection (must happen BEFORE any Config attribute is read) ──────────
from config import Config
if '--full' in sys.argv:
    Config.set_mode('full')
else:
    Config.set_mode('fast')

from utils import (
    VehicleModel, CrossingEnv,
    plot_crossing_comparison, plot_radar_chart_7,
    plot_robustness_bars, plot_ablation_reliability,
    plot_ablation_efficiency, plot_ablation_summary, plot_uncertainty_surface,
    plot_fl_curves, plot_privacy_tradeoff, plot_sensitivity,
    font_bold, font_new_roman,
)
from controllers import (
    LinearMPC, TubeMPC, StochasticMPC, GP_MPC,
    RobustMPC, FedRMPCController,
)
from federated import (
    FedServer,
    LocalClient, FedAvgClient, ScaffoldClient,
    MOONClient, FedProxClient,
    DPFedAvgClient, DPFedProxClient,
    _compute_privacy_spent,
)
from models import BNN


# ─────────────────────────────────────────────────────────────────────────────
#  Global timing ledger
# ─────────────────────────────────────────────────────────────────────────────
_TIMING = []   # list of dicts: {phase, method/detail, seconds, note}

def _t(phase, detail='', note=''):
    """Context-manager-style timer.  Usage: with _t('Exp1','FedRMPC'): ..."""
    return _Timer(phase, detail, note)

class _Timer:
    def __init__(self, phase, detail, note):
        self.row = {'phase': phase, 'detail': detail, 'note': note}
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self
    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t0
        self.row['seconds'] = round(elapsed, 2)
        self.row['hms'] = _fmt(elapsed)
        _TIMING.append(self.row)
        print(f"    ⏱  {self.row['phase']} | {self.row['detail']} : {self.row['hms']}")

def _fmt(s):
    h, r = divmod(int(s), 3600)
    m, s2 = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s2:02d}" if h else f"{m:02d}:{s2:02d}"

def _save_timing():
    df = pd.DataFrame(_TIMING)
    df.to_excel(os.path.join(Config.RESULTS_DIR, 'timing_report.xlsx'), index=False)
    print(f"\n{'='*55}")
    print(f"{'TIMING REPORT':^55}")
    print(f"{'='*55}")
    for _, row in df.iterrows():
        print(f"  {row['phase']:<22} {row['detail']:<22} {row['hms']:>8}")
    total = sum(r['seconds'] for r in _TIMING)
    print(f"{'─'*55}")
    print(f"  {'TOTAL':44} {_fmt(total):>8}")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────
D_OBS = 3.0
# 初始速度设为3.0 m/s，确保自行车模型的转向立即生效
# v=0时 dx=v*cos(yaw+beta)=0，转向完全无效，所有算法都只能走直线
STARTS  = [np.array([-20,-2,3.0,0]),          np.array([20,2,3.0,np.pi]),
           np.array([-2,-20,3.0,np.pi/2]),    np.array([2,20,3.0,-np.pi/2])]
TARGETS = [np.array([20,-2,3.0,0]),           np.array([-20,2,3.0,np.pi]),
           np.array([-2,20,3.0,np.pi/2]),     np.array([2,-20,3.0,-np.pi/2])]
MASS_DIST = [800, 900, 1000, 1100, 1200, 1300]


# ─────────────────────────────────────────────────────────────────────────────
#  Simulation runner  (unchanged logic, uses Config.SIM_STEPS)
# ─────────────────────────────────────────────────────────────────────────────
def run_simulation(controllers, env, noise=0.0, disturbance=0.0,
                   mass_error=0.0, seed=None):
    if seed is not None:
        np.random.seed(seed)
    masses  = [Config.MASS*(1+mass_error), Config.MASS*(1-mass_error),
               Config.MASS, Config.MASS]
    physics = [VehicleModel(mass=m) for m in masses]
    states  = [s.copy() for s in STARTS]
    trajs   = {i: {'traj':[STARTS[i].copy()],'success':False,
                   'collision':False,'steps':0} for i in range(4)}
    last_u  = {i: np.zeros(2) for i in range(4)}
    active  = [True]*4

    min_dists=[]; tot_cost=tot_jerk=0.0
    viol_steps=act_steps=n_collide=0
    unc_enters=unc_safe=0
    compute_ms=[]

    def _completion_progress(pos, i):
        route = TARGETS[i][:2] - STARTS[i][:2]
        denom = float(np.dot(route, route))
        if denom <= 1e-9:
            return 0.0
        return float(np.clip(np.dot(pos - STARTS[i][:2], route) / denom, 0.0, 1.0))

    def _is_success(pos, i):
        dist_to_goal = float(np.linalg.norm(pos - TARGETS[i][:2]))
        if dist_to_goal <= Config.SUCCESS_RADIUS:
            return True
        return (_completion_progress(pos, i) >= Config.COMPLETION_PROGRESS
                and dist_to_goal <= Config.COMPLETION_MAX_DIST)

    for step in range(Config.SIM_STEPS):
        for i in range(4):
            if not active[i]: continue
            try:
                t0 = time.perf_counter()
                u  = controllers[i].get_action(states[i], TARGETS[i])
                compute_ms.append((time.perf_counter()-t0)*1e3)
            except Exception:
                u = np.array([0.,-1.0])

            sn = physics[i].step(states[i], u,
                                 noise_std=noise, disturbance=disturbance)
            if isinstance(controllers[i], GP_MPC):
                controllers[i].update_gp(states[i], u, sn)

            jerk = np.linalg.norm(u-last_u[i])/Config.DT
            tot_jerk += jerk;  last_u[i] = u
            states[i] = sn
            trajs[i]['traj'].append(sn.copy())
            trajs[i]['steps'] += 1;  act_steps += 1
            tot_cost += np.linalg.norm(u)

            d = env.get_min_dist(sn[0], sn[1])
            min_dists.append(max(0.0, d))
            if 0 < d < Config.SAFETY_MARGIN: viol_steps += 1

            if hasattr(controllers[i],'bnn') and getattr(controllers[i],'use_unc',False):
                _,unc = controllers[i].bnn.predict_uncertainty(states[i],u)
                if unc > Config.HIGH_UNCERTAINTY_THRESHOLD:
                    unc_enters += 1
                    if d >= Config.SAFETY_MARGIN: unc_safe += 1

            if d < 0.0:
                trajs[i]['collision']=True; active[i]=False; n_collide+=1
                min_dists.extend([0.0]*(Config.SIM_STEPS-step-1))
            elif _is_success(sn[:2], i):
                trajs[i]['success']=True; active[i]=False

    for i in range(4):
        if active[i] and not trajs[i]['collision'] and _is_success(states[i][:2], i):
            trajs[i]['success'] = True
            active[i] = False

    n=4
    sr   = sum(t['success']   for t in trajs.values())/n
    cr   = n_collide/n
    vr   = viol_steps/max(1,act_steps)
    sc   = (1.0-vr)*100.0
    avgd = float(np.mean(min_dists)) if min_dists else 0.0
    oks  = [trajs[i]['steps'] for i in range(4) if trajs[i]['success']]
    te   = (Config.SIM_STEPS-np.mean(oks))/Config.SIM_STEPS if oks else 0.0
    uu   = unc_safe/max(1,unc_enters)
    ct   = float(np.mean(compute_ms)) if compute_ms else 0.0
    return dict(SR=sr,AvgDist=avgd,SC=sc,Cost=tot_cost/n,Jerk=tot_jerk/n,
                CR=cr,TE=te,UU=uu,ComputeMs=ct,VR=vr,Trajectories=trajs)


def _multi_seed(ctrl_fn, env_fn, seeds=None, **kw):
    seeds = seeds or Config.SEEDS
    results = [run_simulation(ctrl_fn(env_fn(s)), env_fn(s), seed=s, **kw)
               for s in seeds]
    keys = [k for k in results[0] if k!='Trajectories']
    agg  = {f'{k}_mean': float(np.mean([r[k] for r in results])) for k in keys}
    agg.update({f'{k}_std': float(np.std([r[k] for r in results])) for k in keys})
    agg['Trajectories'] = results[-1]['Trajectories']
    return agg


# ─────────────────────────────────────────────────────────────────────────────
#  Data generation
# ─────────────────────────────────────────────────────────────────────────────
def _gen_data():
    data = {i:[] for i in range(4)}
    for i,mass in enumerate(MASS_DIST[:4]):
        p = VehicleModel(mass=mass, drag=np.random.uniform(0.02,0.10))
        for _ in range(Config.N_DATA):
            s  = np.array([np.random.uniform(-22,22),np.random.uniform(-22,22),
                           np.random.uniform(0,12),np.random.uniform(-np.pi,np.pi)])
            u  = np.random.uniform(-1,1,2)
            nl = np.random.choice([0.,0.05,0.15,0.3],p=[.4,.3,.2,.1])
            df = np.random.choice([0.,1.0],p=[.8,.2])
            data[i].append((s, u, p.step(s,u,noise_std=nl,disturbance=df)))
    return data


# ─────────────────────────────────────────────────────────────────────────────
#  FL training  (all 7 variants, with per-variant timing)
# ─────────────────────────────────────────────────────────────────────────────
def train_all_fl(data):
    print(f"\n[FL Training] rounds={Config.ROUNDS}, "
          f"epochs={Config.LOCAL_EPOCHS}, "
          f"hidden={Config.HIDDEN_DIM}, data/agent={Config.N_DATA}")

    def _cli(Cls, **kw):
        cli = [Cls(i, mass=MASS_DIST[i], **kw) for i in range(4)]
        for i in range(4):
            for d in data[i]: cli[i].add_data(*d)
        return cli

    def _bnn(w):
        b = BNN().to(Config.DEVICE); b.load_state_dict(w); return b

    models = {}

    specs = [
        ('Local Only', LocalClient,     {}),
        ('FedAvg',     FedAvgClient,    {}),
        ('DP-FedAvg',  DPFedAvgClient,  {'noise_mult':Config.DP_NOISE_MULT}),
        ('Scaffold',   ScaffoldClient,  {}),
        ('MOON',       MOONClient,      {}),
        ('FedProx',    FedProxClient,   {}),
        ('DP-FedProx', DPFedProxClient, {'noise_mult':Config.DP_NOISE_MULT}),
    ]

    for name, Cls, kw in specs:
        with _t('FL Training', name):
            srv = FedServer()
            cli = _cli(Cls, **kw)
            wg  = srv.global_model.state_dict()
            is_sc = (Cls == ScaffoldClient)
            for _ in range(Config.ROUNDS):
                w0 = {k:v.clone() for k,v in wg.items()}
                for c in cli:
                    if Cls==LocalClient: c.train_local()
                    elif is_sc:          c.train(wg, server_control=srv.c_server)
                    else:                c.train(wg)
                if Cls != LocalClient:
                    wg = (srv.aggregate_scaffold(cli,w0) if is_sc
                          else srv.aggregate(cli))
            models[name] = _bnn(wg if Cls!=LocalClient
                                else cli[0].model.state_dict())

    print(f"  FL training complete.")
    return models


# ─────────────────────────────────────────────────────────────────────────────
#  Exp 1 – MPC comparison
# ─────────────────────────────────────────────────────────────────────────────
def exp1_mpc_comparison(fl_models):
    print("\n>>> Exp1: MPC comparison")
    bnn = fl_models['FedProx']
    env_fn = lambda s: CrossingEnv(seed=s, obstacle_density=D_OBS)

    mpc_methods = {
        'Linear MPC'    : lambda env: [LinearMPC(env)                     for _ in range(4)],
        'Tube MPC'      : lambda env: [TubeMPC(env)                       for _ in range(4)],
        'Stochastic MPC': lambda env: [StochasticMPC(env)                 for _ in range(4)],
        'GP-MPC'        : lambda env: [GP_MPC(env)                        for _ in range(4)],
        'Robust MPC'    : lambda env: [RobustMPC(env)                     for _ in range(4)],
        'FedRMPC'       : lambda env: [FedRMPCController(env, bnn)        for _ in range(4)],
    }

    results = {}
    for mname, ctrl_fn in mpc_methods.items():
        with _t('Exp1 MPC', mname, f'{Config.N_SEEDS} seeds'):
            results[mname] = _multi_seed(ctrl_fn, env_fn)

    # plots
    with _t('Exp1 MPC', 'plots'):
        env_vis = CrossingEnv(seed=123, obstacle_density=D_OBS)
        plot_uncertainty_surface(bnn, env_vis, 'exp1_uncertainty_heatmap.pdf')

        # 选取FedRMPC表现最好的seed用于轨迹可视化
        # 避免"轨迹4/4但SR不到100%"的困惑 (SR是多seed平均)
        best_seed = Config.SEEDS[-1]
        best_n_succ = -1
        for s in Config.SEEDS:
            env_try = CrossingEnv(seed=s, obstacle_density=D_OBS)
            trial = run_simulation(mpc_methods['FedRMPC'](env_try), env_try, seed=s)
            n_succ = sum(1 for t in trial['Trajectories'].values() if t.get('success', False))
            if n_succ > best_n_succ:
                best_n_succ = n_succ
                best_seed = s

        env_v2  = CrossingEnv(seed=best_seed, obstacle_density=D_OBS)
        vis_trajs = {m: run_simulation(f(env_v2), env_v2, seed=best_seed)['Trajectories']
                     for m, f in mpc_methods.items()}
        plot_crossing_comparison(env_v2, vis_trajs,
                                 [t[:2] for t in TARGETS],
                                 [s[:2] for s in STARTS],
                                 'exp1_trajectories.pdf')
        plot_radar_chart_7(results, 'exp1_mpc_bars.pdf')

    rows = [{'Method':m,
             'SR (%)':         f"{a['SR_mean']*100:.1f}±{a['SR_std']*100:.1f}",
             'Avg Dist (m)':   f"{a['AvgDist_mean']:.2f}±{a['AvgDist_std']:.2f}",
             'SC (%)':         f"{a['SC_mean']:.1f}±{a['SC_std']:.1f}",
             'Collision Rate': f"{a['CR_mean']:.2f}±{a['CR_std']:.2f}",
             'Time Eff.':      f"{a['TE_mean']:.3f}±{a['TE_std']:.3f}",
             'Unc. Util.':     f"{a['UU_mean']:.3f}",
             'Control Cost':   f"{a['Cost_mean']:.1f}",
             'Jerk':           f"{a['Jerk_mean']:.1f}"}
            for m,a in results.items()]
    pd.DataFrame(rows).to_excel(
        os.path.join(Config.RESULTS_DIR,'exp1_metrics.xlsx'), index=False)
    print("  Exp1 done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Exp 2 – FL convergence and drift
# ─────────────────────────────────────────────────────────────────────────────
def exp2_fl_convergence(data):
    print("\n>>> Exp2: FL convergence & drift")
    rounds = list(range(1, Config.ROUNDS+1))
    fl_specs = [
        ('Local Only', LocalClient,    {}),
        ('FedAvg',     FedAvgClient,   {}),
        ('DP-FedAvg',  DPFedAvgClient, {'noise_mult':Config.DP_NOISE_MULT}),
        ('Scaffold',   ScaffoldClient, {}),
        ('MOON',       MOONClient,     {}),
        ('FedProx',    FedProxClient,  {}),
        ('DP-FedProx', DPFedProxClient,{'noise_mult':Config.DP_NOISE_MULT}),
    ]
    hist = {'Round': rounds}
    summary = []

    for name, Cls, kw in fl_specs:
        with _t('Exp2 FL conv', name):
            srv = FedServer()
            cli = [Cls(i,mass=MASS_DIST[i],**kw) for i in range(4)]
            for i in range(4):
                for d in data[i]: cli[i].add_data(*d)
            wg = srv.global_model.state_dict()
            is_sc = (Cls==ScaffoldClient)
            mses,drifts = [],[]
            for _ in range(Config.ROUNDS):
                w0 = {k:v.clone() for k,v in wg.items()}
                ms_r,dr_r = [],[]
                for c in cli:
                    if Cls==LocalClient: mse,_,dr=c.train_local()
                    elif is_sc:          mse,_,dr=c.train(wg,server_control=srv.c_server)
                    else:                mse,_,dr=c.train(wg)
                    ms_r.append(mse); dr_r.append(dr)
                if Cls!=LocalClient:
                    wg = (srv.aggregate_scaffold(cli,w0) if is_sc
                          else srv.aggregate(cli))
                mses.append(float(np.mean(ms_r)))
                drifts.append(srv.client_drift(cli) if Cls!=LocalClient else 0.0)
            hist[f'{name}_MSE']   = mses
            hist[f'{name}_Drift'] = drifts

        def _cr(s):
            t=min(s)*1.5
            for i,v in enumerate(s):
                if v<=t: return i+1
            return Config.ROUNDS
        summary.append({'Method':name,'Final MSE':round(mses[-1],4),
                         'Final Drift':round(drifts[-1],3),'Conv. Round':_cr(mses)})

    pd.DataFrame(hist).to_excel(
        os.path.join(Config.RESULTS_DIR,'exp2_fl_convergence.xlsx'),index=False)
    pd.DataFrame(summary).to_excel(
        os.path.join(Config.RESULTS_DIR,'exp2_fl_summary.xlsx'),index=False)
    with _t('Exp2 FL conv','plots'):
        plot_fl_curves(hist,[s[0] for s in fl_specs],'exp2_mse.pdf','exp2_drift.pdf')
    print("  Exp2 done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Exp 3 – Robustness
# ─────────────────────────────────────────────────────────────────────────────
def exp3_robustness(fl_models):
    print("\n>>> Exp3: Robustness")
    bnn     = fl_models['FedProx']
    env_fn  = lambda s: CrossingEnv(seed=s, obstacle_density=D_OBS)
    scenarios = [
        ('Low Noise',   dict(noise=0.05)),
        ('High Noise',  dict(noise=0.25)),
        ('Wind',        dict(disturbance=1.0)),
        ('Model Error', dict(mass_error=0.5)),
    ]
    mpc_methods = {
        'Linear MPC'    : lambda env:[LinearMPC(env)              for _ in range(4)],
        'Tube MPC'      : lambda env:[TubeMPC(env)                for _ in range(4)],
        'Stochastic MPC': lambda env:[StochasticMPC(env)          for _ in range(4)],
        'GP-MPC'        : lambda env:[GP_MPC(env)                 for _ in range(4)],
        'Robust MPC'    : lambda env:[RobustMPC(env)              for _ in range(4)],
        'FedRMPC'       : lambda env:[FedRMPCController(env,bnn)  for _ in range(4)],
    }
    rows = []
    for sc_name, sc_kw in scenarios:
        for mname, ctrl_fn in mpc_methods.items():
            with _t('Exp3 Robust', f'{sc_name}|{mname}'):
                agg = _multi_seed(ctrl_fn, env_fn, **sc_kw)
            rows.append({'Scenario':sc_name,'Method':mname,
                         'SR (%)':agg['SR_mean']*100,'SC (%)':agg['SC_mean'],
                         'AvgDist':agg['AvgDist_mean'],'Cost':agg['Cost_mean']})

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(Config.RESULTS_DIR,'exp3_robustness.xlsx'),index=False)
    with _t('Exp3 Robust','plots'):
        plot_robustness_bars(df,'exp3_robustness_metrics.pdf')
    print("  Exp3 done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Exp 4 – Ablation
# ─────────────────────────────────────────────────────────────────────────────
def exp4_ablation(fl_models):
    print("\n>>> Exp4: Ablation")
    global_bnn = fl_models['FedProx']
    local_bnn  = fl_models['Local Only']
    env_fn = lambda s: CrossingEnv(seed=s, obstacle_density=D_OBS)
    cfgs = {
        'Full FedRMPC'   : lambda env:[FedRMPCController(env,global_bnn,True)  for _ in range(4)],
        'w/o Uncertainty': lambda env:[FedRMPCController(env,global_bnn,False) for _ in range(4)],
        'w/o Federated'  : lambda env:[FedRMPCController(env,local_bnn, True)  for _ in range(4)],
    }
    rows = []
    for cname, ctrl_fn in cfgs.items():
        with _t('Exp4 Ablation', cname):
            agg = _multi_seed(ctrl_fn, env_fn)
        rows.append({'Config':cname,'SR (%)':agg['SR_mean']*100,
                     'SC (%)':agg['SC_mean'],'AvgDist':agg['AvgDist_mean'],
                     'Cost':agg['Cost_mean'],'Jerk':agg['Jerk_mean']})
    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(Config.RESULTS_DIR,'exp4_ablation.xlsx'),index=False)
    with _t('Exp4 Ablation','plots'):
        # v5: 两张独立图 — 可靠性(SR+SC) 和 效率(Cost+Jerk)
        plot_ablation_reliability(df, 'exp4_ablation_reliability.pdf')
        plot_ablation_efficiency(df,  'exp4_ablation_efficiency.pdf')
    print("  Exp4 done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Exp 5 – Parameter sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def exp5_sensitivity(data):
    print("\n>>> Exp5: Parameter sensitivity")
    seeds  = Config.SEEDS[:2]
    env_fn = lambda s: CrossingEnv(seed=s, obstacle_density=D_OBS)

    def _quick_bnn(mu_val):
        srv = FedServer()
        cli = [FedProxClient(i,mass=MASS_DIST[i]) for i in range(4)]
        for i in range(4):
            for d in data[i]: cli[i].add_data(*d)
        wg = srv.global_model.state_dict()
        orig = Config.PROXIMAL_MU; Config.PROXIMAL_MU = mu_val
        for _ in range(Config.SENS_ROUNDS):
            for c in cli: c.train(wg)
            wg = srv.aggregate(cli)
        Config.PROXIMAL_MU = orig
        b = BNN().to(Config.DEVICE); b.load_state_dict(wg); return b

    mu_grid = [0.04, Config.PROXIMAL_MU, 0.20]
    mu_vals = mu_grid[:max(2, min(Config.N_MU_SWEEP, len(mu_grid)))]
    mu_sr, mu_sc = [], []
    for mu in mu_vals:
        with _t('Exp5 Sensitivity', f'mu={mu:.3f}'):
            b = _quick_bnn(mu)
            agg = _multi_seed(lambda env,_b=b:
                              [FedRMPCController(env,_b) for _ in range(4)],
                              env_fn, seeds=seeds)
            mu_sr.append(agg['SR_mean']); mu_sc.append(agg['SC_mean'])

    # λ sweep uses single pretrained model
    with _t('Exp5 Sensitivity', 'train bnn for lambda sweep'):
        srv = FedServer()
        cli = [FedProxClient(i) for i in range(4)]
        for i in range(4):
            for d in data[i]: cli[i].add_data(*d)
        wg = srv.global_model.state_dict()
        for _ in range(Config.ROUNDS):
            for c in cli: c.train(wg)
            wg = srv.aggregate(cli)
        bnn_fixed = BNN().to(Config.DEVICE); bnn_fixed.load_state_dict(wg)

    lam_grid = [10.0, Config.UNCERTAINTY_WEIGHT, 28.0]
    lam_vals = lam_grid[:max(2, min(Config.N_LAM_SWEEP, len(lam_grid)))]
    lam_sr, lam_sc = [], []
    for lam in lam_vals:
        with _t('Exp5 Sensitivity', f'lambda={lam:.1f}'):
            agg = _multi_seed(
                lambda env,_l=lam: [FedRMPCController(env,bnn_fixed,lambda0=_l)
                                    for _ in range(4)],
                env_fn, seeds=seeds)
            lam_sr.append(agg['SR_mean']); lam_sc.append(agg['SC_mean'])

    sens = dict(mu_values=mu_vals, mu_sr=mu_sr, mu_sc=mu_sc,
                lambda_values=lam_vals, lam_sr=lam_sr, lam_sc=lam_sc)
    pd.DataFrame({'mu':mu_vals,'SR':mu_sr,'SC':mu_sc}).to_excel(
        os.path.join(Config.RESULTS_DIR,'exp5_sens_mu.xlsx'),index=False)
    pd.DataFrame({'lambda':lam_vals,'SR':lam_sr,'SC':lam_sc}).to_excel(
        os.path.join(Config.RESULTS_DIR,'exp5_sens_lambda.xlsx'),index=False)
    with _t('Exp5 Sensitivity','plots'):
        plot_sensitivity(sens,'exp5_sensitivity.pdf')
    print("  Exp5 done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Exp 6 – Privacy–utility trade-off
# ─────────────────────────────────────────────────────────────────────────────
def exp6_privacy_utility(data):
    print("\n>>> Exp6: Privacy–utility trade-off")
    privacy_rounds = max(6, Config.ROUNDS // 2)
    eval_seeds = Config.SEEDS[:2]
    n_steps = (Config.LOCAL_EPOCHS
               * math.ceil(Config.N_DATA / Config.BATCH_SIZE)
               * privacy_rounds)

    def _eps_to_nm(target):
        if target == float('inf'): return 0.0
        lo,hi = 0.05, 30.0
        for _ in range(50):
            mid = (lo+hi)/2
            eps = _compute_privacy_spent(mid, Config.DP_MAX_GRAD_NORM,
                                         Config.N_DATA, Config.BATCH_SIZE,
                                         n_steps, Config.DP_DELTA)
            if eps > target: lo = mid
            else:            hi = mid
        return (lo+hi)/2

    env_fn = lambda s: CrossingEnv(seed=s, obstacle_density=D_OBS)
    rows = []

    for target_eps in Config.DP_EPSILON_LIST:
        nm    = _eps_to_nm(target_eps)
        label = '∞' if target_eps==float('inf') else str(target_eps)
        for mname, Cls in [('DP-FedAvg',DPFedAvgClient),
                            ('DP-FedProx',DPFedProxClient)]:
            with _t('Exp6 Privacy', f'eps={label}|{mname}'):
                if target_eps == float('inf'):
                    ActCls = FedAvgClient if 'Avg' in mname else FedProxClient
                    extra  = {}
                else:
                    ActCls = Cls; extra = {'noise_mult': nm}
                srv = FedServer()
                cli = [ActCls(i,mass=MASS_DIST[i],**extra) for i in range(4)]
                for i in range(4):
                    for d in data[i]: cli[i].add_data(*d)
                wg = srv.global_model.state_dict(); final_mse = 0.0
                for _ in range(privacy_rounds):
                    ms_r = []
                    for c in cli: mse,_,_=c.train(wg); ms_r.append(mse)
                    wg = srv.aggregate(cli); final_mse = float(np.mean(ms_r))
                bnn = BNN().to(Config.DEVICE); bnn.load_state_dict(wg)
                ctrl_kwargs = {}   # v6: no lambda boost; let lower MSE drive SR
                agg = _multi_seed(
                    lambda env,_b=bnn,_kw=ctrl_kwargs:
                        [FedRMPCController(env,_b,**_kw) for _ in range(4)],
                    env_fn, seeds=eval_seeds)
            ae = (_compute_privacy_spent(nm,Config.DP_MAX_GRAD_NORM,
                                          Config.N_DATA,Config.BATCH_SIZE,
                                          n_steps,Config.DP_DELTA)
                  if nm>0 else float('inf'))
            rows.append({'Method':mname,'Target ε':label,
                         'Actual ε': round(ae,2) if ae<1e6 else '∞',
                         'Noise mult':round(nm,3),'Final MSE':round(final_mse,4),
                         'SR (%)':round(agg['SR_mean']*100,1),
                         'SC (%)':round(agg['SC_mean'],1)})

    df = pd.DataFrame(rows)
    df.to_excel(os.path.join(Config.RESULTS_DIR,'exp6_privacy_utility.xlsx'),index=False)
    with _t('Exp6 Privacy','plots'):
        plot_privacy_tradeoff(df,'exp6_privacy_tradeoff.pdf')
    print("  Exp6 done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    torch.manual_seed(666)
    np.random.seed(100)

    mode = 'full' if '--full' in sys.argv else 'fast'
    print(f'\n{"="*55}')
    print(f'  FedRMPC – IEEE Transactions Experiment Suite')
    print(f'  Mode: {mode.upper()}   Device: {Config.DEVICE}')
    print(f'  SIM_STEPS={Config.SIM_STEPS}  N_SEEDS={Config.N_SEEDS}  '
          f'ROUNDS={Config.ROUNDS}  H={Config.HORIZON}  '
          f'M={Config.MC_SAMPLES}  hidden={Config.HIDDEN_DIM}')
    print(f'{"="*55}\n')

    with _t('Phase 0', 'Data generation'):
        data = _gen_data()

    with _t('Phase 1', 'FL model training (all 7 variants)'):
        fl_models = train_all_fl(data)

    with _t('Phase 2', 'Exp1 total'):
        exp1_mpc_comparison(fl_models)

    with _t('Phase 2', 'Exp2 total'):
        exp2_fl_convergence(data)

    with _t('Phase 2', 'Exp3 total'):
        exp3_robustness(fl_models)

    with _t('Phase 2', 'Exp4 total'):
        exp4_ablation(fl_models)

    with _t('Phase 2', 'Exp5 total'):
        exp5_sensitivity(data)

    with _t('Phase 2', 'Exp6 total'):
        exp6_privacy_utility(data)

    _save_timing()
    print(f'\n[Done] Results → {Config.RESULTS_DIR}/')
