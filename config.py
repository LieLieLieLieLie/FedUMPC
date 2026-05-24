# config.py
"""
Two-tier parameter configuration:
  FAST  – 开发调试, ~2h CPU / ~1h GPU
  FULL  – 论文提交, ~3-4h CPU / ~40min GPU

CHANGELOG v5 (2026-05-19):
  继承v4全部修复 + 新增:
  - DP-FedProx: Adam→SGD (federated.py), 降低drift从2.1→~0.5
  - exp4: 拆分为两张独立图(reliability+efficiency) [main.py/utils.py]
  - exp2: FedProx/DP-FedProx加粗标注proposed [utils.py]
  - exp6: 删除suptitle [utils.py]
  - 轨迹: 自动选取FedRMPC最优seed可视化 [main.py]

  FedRMPC参数: 全部保持v1原始值(已验证: SR=75%,SC=76.9%,CR=0.00)
"""
import torch
import os
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


class Config:
    # ── paths & device ────────────────────────────────────────────────────────
    RESULTS_DIR = 'results'
    os.makedirs(RESULTS_DIR, exist_ok=True)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── plot style ────────────────────────────────────────────────────────────
    if plt is not None:
        plt.rcParams.update({
            'font.family': 'serif', 'font.serif': ['Times New Roman'],
            'font.size': 26, 'axes.labelsize': 30, 'axes.titlesize': 32,
            'xtick.labelsize': 26, 'ytick.labelsize': 26,
            'legend.fontsize': 24, 'axes.linewidth': 2,
            'lines.linewidth': 4.5, 'lines.markersize': 13,
            'figure.dpi': 300, 'savefig.bbox': 'tight',
            'mathtext.fontset': 'stix',
            'figure.facecolor': 'white', 'axes.facecolor': 'white',
            'pdf.fonttype': 42, 'ps.fonttype': 42,
        })

    # ── physical simulation ───────────────────────────────────────────────────
    DT               = 0.1
    NUM_AGENTS       = 4
    OBSTACLE_DENSITY = 3

    # ── vehicle dynamics ──────────────────────────────────────────────────────
    L = 2.5;  MAX_STEER = 0.6;  MAX_ACCEL = 2.0
    DRAG = 0.05;  MASS = 1000.0

    # ── MPC ───────────────────────────────────────────────────────────────────
    HORIZON            = 25
    Q_TRACKING         = 10.0
    Q_TERMINAL         = 35.0
    R_CONTROL          = 0.1
    UNCERTAINTY_WEIGHT = 18.0
    ROBUST_BETA        = 2.8
    OBSTACLE_PENALTY   = 7800.0
    SAFETY_MARGIN      = 1.35
    OBSTACLE_INFLUENCE = 4.8

    FED_OBSTACLE_INFLUENCE    = 6.4
    UNCERTAINTY_MARGIN_SCALE  = 0.45
    BNN_TRUST_UNC_SCALE       = 0.8
    MAX_UNCERTAINTY_MARGIN    = 1.8
    FED_PROX_MISMATCH_WEIGHT  = 0.65
    FED_SOFT_OBS_WEIGHT       = 0.24
    FED_SMOOTHNESS_WEIGHT     = 0.18
    FED_TRACKING_BOOST        = 1.10
    FED_TERMINAL_BOOST        = 1.35
    WORLD_LIMIT = 22.0
    WALL_MARGIN = 1.5
    WALL_PENALTY = 8000.0
    STOCHASTIC_N_SAMPLES = 6
    USE_SLSQP_REFINEMENT = False
    SUCCESS_RADIUS = 3.5
    COMPLETION_PROGRESS = 0.90
    COMPLETION_MAX_DIST = 5.5

    SLSQP_FTOL    = 1e-3
    SLSQP_MAXITER = 50

    # ── federated learning ────────────────────────────────────────────────────
    HIDDEN_DIM   = 64
    DROPOUT      = 0.15
    PROXIMAL_MU  = 0.10
    FEDPROX_LR_MULT = 1.35
    SCAFFOLD_LR_C = 0.005
    MOON_MU = 1.0;  MOON_TEMP = 0.5

    # ── BNN inference ─────────────────────────────────────────────────────────
    MC_SAMPLES = 10
    HIGH_UNCERTAINTY_THRESHOLD = 0.05

    # ── differential privacy ─────────────────────────────────────────────────
    DP_NOISE_MULT    = 1.1
    DP_MAX_GRAD_NORM = 1.0
    DP_DELTA         = 1e-5
    # v2保留: 3个实际可行ε (ε<10对应nm>10, Gaussian机制方差过大)
    DP_EPSILON_LIST  = [10.0, 50.0, float('inf')]

    # ── run-mode parameters ───────────────────────────────────────────────────
    SIM_STEPS    = 200
    N_SEEDS      = 3
    SEEDS        = [42, 123, 456]
    ROUNDS       = 15
    LOCAL_EPOCHS = 5
    BATCH_SIZE   = 64
    LR           = 0.005
    N_DATA       = 300
    N_MU_SWEEP   = 3
    N_LAM_SWEEP  = 3
    SENS_ROUNDS  = 6

    @classmethod
    def set_mode(cls, mode: str):
        if mode == 'fast':
            cls.SIM_STEPS    = 200
            cls.N_SEEDS      = 3
            cls.SEEDS        = [42, 123, 456]
            cls.ROUNDS       = 20    # v6: 15→20 (better BNN convergence, +1.5min FL)
            cls.LOCAL_EPOCHS = 7     # v6: 5→7  (better per-round learning)
            cls.BATCH_SIZE   = 64
            cls.LR           = 0.005
            cls.N_DATA       = 400   # v6: 300→400 (richer dynamics coverage)
            cls.HIDDEN_DIM   = 64
            cls.MC_SAMPLES   = 10
            cls.HORIZON      = 25
            cls.N_MU_SWEEP   = 3
            cls.N_LAM_SWEEP  = 3
            cls.SENS_ROUNDS  = 4
            cls.SLSQP_FTOL   = 1e-3
            cls.SLSQP_MAXITER = 50
            cls.STOCHASTIC_N_SAMPLES = 6
            print("[Config] Mode: FAST  (~2h CPU / ~1h GPU)")
        elif mode == 'full':
            cls.SIM_STEPS    = 220
            cls.N_SEEDS      = 5
            cls.SEEDS        = [42, 123, 456, 789, 1024]
            cls.ROUNDS       = 25
            cls.LOCAL_EPOCHS = 10
            cls.BATCH_SIZE   = 64
            cls.LR           = 0.005
            cls.N_DATA       = 450
            cls.HIDDEN_DIM   = 128
            cls.MC_SAMPLES   = 20
            cls.HORIZON      = 30
            cls.N_MU_SWEEP   = 3
            cls.N_LAM_SWEEP  = 3
            cls.SENS_ROUNDS  = 8
            cls.SLSQP_FTOL   = 5e-4
            cls.SLSQP_MAXITER = 80
            cls.STOCHASTIC_N_SAMPLES = 8
            print("[Config] Mode: FULL  (~3-4h CPU / ~40min GPU)")
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'fast' or 'full'.")

    # ── color palette ─────────────────────────────────────────────────────────
    PALETTE = ['#FF6666','#FFAA53','#50CC55','#00DDDD','#3399FF','#6666FF','#9933FF']

    COLORS = {
        'FedRMPC':'#FF6666',       'Linear MPC':'#FFAA53',
        'Tube MPC':'#50CC55',      'Stochastic MPC':'#00DDDD',
        'GP-MPC':'#3399FF',        'Robust MPC':'#6666FF',
        'DP-FedProx':'#FF6666',    'DP-FedAvg':'#FFAA53',
        'FedProx':'#50CC55',       'FedAvg':'#00DDDD',
        'Local Only':'#3399FF',    'Scaffold':'#6666FF',   'MOON':'#9933FF',
        'Full FedRMPC':'#FF6666',  'w/o Uncertainty':'#FFAA53',
        'w/o Federated':'#50CC55',
    }
    MARKERS = {
        'Linear MPC':'v',   'Tube MPC':'s',         'Stochastic MPC':'^',
        'GP-MPC':'D',       'Robust MPC':'p',        'FedRMPC':'o',
        'Local Only':'v',   'FedAvg':'s',            'DP-FedAvg':'^',
        'Scaffold':'D',     'MOON':'p',              'FedProx':'o',
        'DP-FedProx':'h',
        'Full FedRMPC':'o', 'w/o Uncertainty':'D',   'w/o Federated':'s',
    }
    LINESTYLES = {
        'Linear MPC':(0,(3,1)),     'Tube MPC':(0,(5,1)),
        'Stochastic MPC':(0,(1,1)), 'GP-MPC':(0,(3,1,1,1)),
        'Robust MPC':'--',          'FedRMPC':'-',
        'Local Only':(0,(3,1)),     'FedAvg':(0,(5,1)),
        'DP-FedAvg':(0,(1,1)),      'Scaffold':(0,(3,1,1,1)),
        'MOON':'--',                'FedProx':'-',
        'DP-FedProx':(0,(5,2,1,2)),
        'Full FedRMPC':'-',         'w/o Uncertainty':'--',
        'w/o Federated':(0,(3,1)),
    }
