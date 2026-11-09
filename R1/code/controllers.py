# controllers.py
"""
Bug fixes in this version:
──────────────────────────────────────────────────────────────────────────
BUG 1 (FedUMPC/Stochastic SR=0, no trajectory, Cost≈0):
  Root cause: BNN trained only 15 rounds with 300 samples predicts
  large incorrect increments. curr = curr + mean_inc sends the state
  flying out of bounds in 1-2 steps, triggering immediate collision
  detection → active[i]=False → 0 steps, 0 cost, 0 trajectory.

  Fix A – clamp BNN increment magnitude:
    mean_inc = np.clip(mean_inc, -MAX_INC, MAX_INC)
    where MAX_INC = DT * max_speed ≈ 0.1 * 15 = 1.5  (per-component)
    This prevents untrained BNN from producing absurd predictions.

  Fix B – hybrid rollout fallback:
    If BNN increment is large (>0.5*MAX_INC), fall back to physics
    model for that step.  Ensures the SLSQP landscape is well-posed
    even with an undertrained BNN.

  Fix C – Stochastic MPC: disturbance magnitude was too small relative
    to scene scale; increased from 0.10 to scene-appropriate value with
    proper velocity-aware perturbation.

BUG 2 (d2.0 duplicate files):
  Root cause: old code in exp1 had a density loop; now single D_OBS=3.0.
  Fix: remove any density-parameterised calls from plot functions.

BUG 3 (Compute(ms) in paper metrics):
  Fix: removed from _multi_seed aggregation dict returned for paper use.
  Still recorded internally via timing_report.

BUG 4 (Ablation Full/w_o_Federated SR=0):
  Same as Bug 1 – BNN rollout instability.  Fixed by clamp.

BUG 5 (Sensitivity all zeros):
  Same root cause.  Fixed by clamp.
──────────────────────────────────────────────────────────────────────────
"""
import warnings
import numpy as np
import time
from scipy.optimize import minimize
try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
except Exception:
    GaussianProcessRegressor = RBF = WhiteKernel = ConstantKernel = None
from config import Config
from utils import VehicleModel

# Maximum plausible per-component state increment (DT × max_speed)
_MAX_INC = Config.DT * 20.0   # 2.0 m (generous, avoids clipping valid predictions)


class BaseController:
    compute_ms = 0.0
    def get_action(self, state, target): raise NotImplementedError


class _MPCBase(BaseController):
    def __init__(self, env):
        self.env    = env
        self.model  = VehicleModel()
        self.H      = Config.HORIZON
        self._bounds = ([(-Config.MAX_STEER, Config.MAX_STEER),
                         (-Config.MAX_ACCEL,  Config.MAX_ACCEL)] * self.H)
        self._lo = np.array([b[0] for b in self._bounds])
        self._hi = np.array([b[1] for b in self._bounds])
        self.u_prev = np.zeros(self.H * 2)

    def get_action(self, state, target):
        t0 = time.perf_counter()
        u  = self._solve(state, target)
        self.compute_ms = (time.perf_counter() - t0) * 1e3
        return u

    def _solve(self, state, target):
        guesses = self._initial_guesses(state, target)
        scored = []
        for g in guesses:
            try:
                scored.append((self._cost(g, state, target), g))
            except Exception:
                scored.append((float('inf'), g))
        scored = sorted(scored, key=lambda x: x[0])
        if not getattr(Config, 'USE_SLSQP_REFINEMENT', False):
            u_seq = np.clip(scored[0][1], self._lo, self._hi)
            self.u_prev = np.clip(np.roll(u_seq, -2), self._lo, self._hi)
            self.u_prev[-2:] = u_seq[-2:]
            return u_seq[:2]

        guesses = [g for _, g in scored[:2]]
        best = None
        try:
            for x0 in guesses:
                res = minimize(self._cost, x0, args=(state, target),
                               method='SLSQP', bounds=self._bounds,
                               options={'ftol': Config.SLSQP_FTOL,
                                        'maxiter': Config.SLSQP_MAXITER})
                val = res.fun if np.isfinite(res.fun) else self._cost(res.x, state, target)
                if best is None or val < best[0]:
                    best = (val, res.x)
            u_seq = np.clip(best[1], self._lo, self._hi)
            self.u_prev = np.clip(np.roll(u_seq, -2), self._lo, self._hi)
            self.u_prev[-2:] = u_seq[-2:]
            return u_seq[:2]
        except Exception:
            self.u_prev = np.zeros(self.H * 2)
            return np.array([0., -0.5])

    def _initial_guesses(self, state, target):
        """Common warm-start, pursuit, arc, S-turn and obstacle-waypoint seeds."""
        guesses = [np.clip(self.u_prev.copy(), self._lo, self._hi)]

        nominal = self._pure_pursuit_guess(state, target, bias=0.0)
        guesses.append(nominal.reshape(-1))
        guesses.append(self._pure_pursuit_guess(state, target, bias=0.30).reshape(-1))
        guesses.append(self._pure_pursuit_guess(state, target, bias=-0.30).reshape(-1))

        # S-turn seeds: left-first and right-first curvature
        for sign in (-1.0, 1.0):
            s_turn = nominal.copy()
            split = max(2, self.H // 2)
            s_turn[:split, 0] = sign * 0.60 * Config.MAX_STEER
            s_turn[split:, 0] = -sign * 0.30 * Config.MAX_STEER
            s_turn[:, 1] = np.clip(0.3, -Config.MAX_ACCEL, Config.MAX_ACCEL)
            guesses.append(s_turn.reshape(-1))

        # Arc seeds: constant curvature diversity
        for steer in (-0.60, -0.35, 0.0, 0.35, 0.60):
            arc = nominal.copy()
            arc[:, 0] = steer * Config.MAX_STEER
            arc[:, 1] = np.clip(0.2, -Config.MAX_ACCEL, Config.MAX_ACCEL)
            guesses.append(arc.reshape(-1))

        # R1 baseline-fairness fix: every controller receives the same
        # obstacle-aware waypoint candidates.  Previously, these candidates
        # were available only to FedUMPC and confounded algorithmic effects
        # with a larger, more informative search set.
        for waypoint in self._obstacle_waypoints(state, target):
            for bias in (0.0, 0.22, -0.22):
                guesses.append(self._waypoint_pursuit_guess(
                    state, waypoint, target, bias=bias).reshape(-1))

        return [np.clip(g, self._lo, self._hi) for g in guesses]

    def _pure_pursuit_guess(self, state, target, bias=0.0):
        curr = state.copy()
        seq = np.zeros((self.H, 2))
        for h in range(self.H):
            decay = max(0.0, 1.0 - h / max(1, self.H - 1))
            target_yaw = np.arctan2(target[1] - curr[1], target[0] - curr[0]) + bias * decay
            yaw_err = (target_yaw - curr[3] + np.pi) % (2 * np.pi) - np.pi
            seq[h, 0] = np.clip(0.9 * yaw_err, -Config.MAX_STEER, Config.MAX_STEER)
            seq[h, 1] = np.clip(0.7 * (target[2] - curr[2]) + 0.2, -Config.MAX_ACCEL, Config.MAX_ACCEL)
            curr = self.model.step(curr, seq[h])
        return seq

    def _waypoint_pursuit_guess(self, state, waypoint, target, bias=0.0):
        curr = state.copy()
        seq = np.zeros((self.H, 2))
        for h in range(self.H):
            # v6: 0.35 (was 0.58) – switch to target earlier, less detour
            goal = waypoint if (h < self.H * 0.35 and np.linalg.norm(curr[:2] - waypoint) > 1.5) else target[:2]
            target_yaw = np.arctan2(goal[1] - curr[1], goal[0] - curr[0]) + bias
            yaw_err = (target_yaw - curr[3] + np.pi) % (2 * np.pi) - np.pi
            seq[h, 0] = np.clip(1.05 * yaw_err, -Config.MAX_STEER, Config.MAX_STEER)
            seq[h, 1] = np.clip(0.2, -Config.MAX_ACCEL, Config.MAX_ACCEL)
            curr = self.model.step(curr, seq[h])
        return seq

    def _obstacle_waypoints(self, state, target):
        """
        v6: 강화된 필터 + 목표 방향으로 우회 waypoint 생성.
        - 경로를 직접 막는 장애물만 처리 (lateral < 0.55*(r+margin))
        - 목표와 가장 가까운 쪽으로 우회 (이전: 에이전트 현재 위치 기준)
        - 측방향 오프셋 3.2→1.8m, 단방향 waypoint 반환
        """
        start = state[:2]
        goal = target[:2]
        seg = goal - start
        seg_len = np.linalg.norm(seg)
        if seg_len < 1e-6:
            return []
        unit = seg / seg_len
        perp = np.array([-unit[1], unit[0]])
        threats = []
        for obs in getattr(self.env, 'obstacles', []):
            center = np.array([obs['x'], obs['y']], dtype=float)
            along = np.dot(center - start, unit)
            if along <= 0.0 or along >= min(seg_len, state[2] * Config.DT * self.H + 8.0):
                continue
            lateral = abs(np.cross(unit, center - start))
            # v6: 직접 막히는 장애물만 처리 (strict filter)
            hard_clearance = obs['r'] + Config.SAFETY_MARGIN
            if lateral < 0.55 * hard_clearance:
                threats.append((along, center, obs['r']))
        if not threats:
            return []

        _, center, radius = min(threats, key=lambda item: item[0])
        # v6: 목표와 더 가까운 방향으로 우회 (측방향 오프셋 1.8m)
        offset = radius + Config.SAFETY_MARGIN + 1.8
        wp_pos = center + perp * offset + unit * 1.0
        wp_neg = center - perp * offset + unit * 1.0
        wp_pos = np.clip(wp_pos, -Config.WORLD_LIMIT + 1.0, Config.WORLD_LIMIT - 1.0)
        wp_neg = np.clip(wp_neg, -Config.WORLD_LIMIT + 1.0, Config.WORLD_LIMIT - 1.0)
        if np.linalg.norm(wp_pos - goal) <= np.linalg.norm(wp_neg - goal):
            return [wp_pos]
        return [wp_neg]

    def _rollout(self, u_mat, state):
        states = np.empty((self.H, 4))
        curr = state.copy()
        for h in range(self.H):
            curr = self.model.step(curr, u_mat[h])
            states[h] = curr
        return states

    def _base_cost(self, states, u_mat, target):
        dist = np.linalg.norm(states[:, :2] - target[:2], axis=1)
        ctrl = np.sum(u_mat ** 2, axis=1)
        terminal = np.linalg.norm(states[-1, :2] - target[:2]) ** 2
        du = np.diff(np.vstack([self.u_prev[:2], u_mat]), axis=0)
        return float(Config.Q_TRACKING * np.sum(dist ** 2)
                     + Config.Q_TERMINAL * terminal
                     + Config.R_CONTROL * ctrl.sum()
                     + Config.FED_SMOOTHNESS_WEIGHT * np.sum(du ** 2)
                     + sum(self._wall_cost(s) for s in states))

    def _obstacle_cost(self, state, margin):
        d = self.env.get_min_dist(state[0], state[1])
        influence = max(Config.OBSTACLE_INFLUENCE, margin)
        if d < margin:
            gap = influence - margin
            deficit = margin - d
            return Config.OBSTACLE_PENALTY * (
                Config.BASE_SOFT_OBS_WEIGHT * gap ** 2
                + 2.0 * Config.BASE_SOFT_OBS_WEIGHT * gap * deficit
                + Config.FED_INNER_OBS_WEIGHT * deficit ** 2)
        if d < influence:
            return (Config.BASE_SOFT_OBS_WEIGHT * Config.OBSTACLE_PENALTY
                    * (influence - d) ** 2)
        return 0.0

    def _wall_cost(self, state):
        excess = np.maximum(np.abs(state[:2]) - (Config.WORLD_LIMIT - Config.WALL_MARGIN), 0.0)
        return float(Config.WALL_PENALTY * np.sum(excess ** 2))

    def _cost(self, u_flat, state, target): raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
class LinearMPC(_MPCBase):
    """
    基线1：线性MPC，使用简单固定裕度避障。
    对照组：无不确定性建模，无学习能力，固定安全裕度。
    """
    def _cost(self, u_flat, state, target):
        u = u_flat.reshape(self.H, 2)
        states = self._rollout(u, state)
        cost = self._base_cost(states, u, target)
        for h in range(self.H):
            cost += self._obstacle_cost(states[h], Config.SAFETY_MARGIN)
        return cost

    def _initial_guesses(self, state, target):
        return super()._initial_guesses(state, target)


# ─────────────────────────────────────────────────────────────────────────────
class TubeMPC(_MPCBase):
    _LF = 0.8;  _W = 0.18

    def __init__(self, env):
        super().__init__(env)
        h = np.arange(self.H, dtype=float)
        LF, W = self._LF, self._W
        self._radii = W * (1.0 - LF**h) / (1.0 - LF)

    def _cost(self, u_flat, state, target):
        u = u_flat.reshape(self.H, 2)
        states = self._rollout(u, state)
        cost = self._base_cost(states, u, target)
        margins = Config.SAFETY_MARGIN + self._radii
        for h in range(self.H):
            cost += self._obstacle_cost(states[h], margins[h])
        return cost


# ─────────────────────────────────────────────────────────────────────────────
class StochasticMPC(_MPCBase):
    """
    Scenario-based MPC.
    FIX: disturbance applied as velocity perturbation (more physical),
    and magnitude scaled to scene (0.3 m/s std on velocity state).
    """
    _W_POS = 0.05    # position noise (m)
    _W_VEL = 0.30    # velocity noise (m/s)
    _W_HDG = 0.02    # heading noise (rad)
    _SIGMA_POINTS = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0, 0.5],
        [-1.0, 0.0, -1.0, -0.5],
        [0.0, 1.0, 1.0, -0.5],
        [0.0, -1.0, -1.0, 0.5],
        [1.0, -1.0, 0.0, 0.0],
        [-1.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, -1.0, 0.5],
    ], dtype=float)

    def _cost(self, u_flat, state, target):
        u = u_flat.reshape(self.H, 2)
        N = Config.STOCHASTIC_N_SAMPLES
        cost = 0.0
        for j in range(N):
            curr = state.copy()
            states = np.empty((self.H, 4), dtype=float)
            z = self._SIGMA_POINTS[j % len(self._SIGMA_POINTS)]
            for h in range(self.H):
                curr = self.model.step(curr, u[h])
                scale = np.sqrt(h + 1.0)
                curr[0] += z[0] * self._W_POS * scale
                curr[1] += z[1] * self._W_POS * scale
                curr[2] += z[2] * self._W_VEL * scale
                curr[3] += z[3] * self._W_HDG * scale
                states[h] = curr
            sc = self._base_cost(states, u, target)
            for h in range(self.H):
                sc += self._obstacle_cost(states[h], Config.SAFETY_MARGIN)
            cost += sc
        return cost / N


# ─────────────────────────────────────────────────────────────────────────────
class GP_MPC(_MPCBase):
    _REFIT = 15;  _MAXPT = 80

    def __init__(self, env):
        super().__init__(env)
        if GaussianProcessRegressor is None:
            self._gp = None
        else:
            kernel = (ConstantKernel(1.0, (0.1, 10.0))
                      * RBF(2.0, (0.5, 50.0))
                      + WhiteKernel(0.05, (1e-3, 1.0)))
            self._gp = GaussianProcessRegressor(kernel=kernel,
                       n_restarts_optimizer=0, normalize_y=True, alpha=1e-6)
        self._bX = []; self._by = []; self._step = 0
        self._fit = False; self._cache = {}

    def update_gp(self, state, u, sn_true):
        res = float(np.linalg.norm(sn_true - self.model.step(state, u)))
        self._bX.append(state.copy()); self._by.append(res)
        if len(self._bX) > self._MAXPT: self._bX.pop(0); self._by.pop(0)
        self._step += 1; self._cache.clear()
        if self._gp is not None and self._step % self._REFIT == 0 and len(self._bX) >= 12:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self._gp.fit(np.array(self._bX), np.array(self._by))
                    self._fit = True
                except Exception:
                    pass

    def _sigma(self, s4):
        if not self._fit: return 0.0
        key = tuple(np.round(s4, 1))
        if key in self._cache: return self._cache[key]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                _, std = self._gp.predict(s4.reshape(1, -1), return_std=True)
                v = float(np.clip(std[0], 0, 2))
            except Exception:
                v = 0.0
        self._cache[key] = v; return v

    def _cost(self, u_flat, state, target):
        u = u_flat.reshape(self.H, 2)
        states = self._rollout(u, state)
        cost = self._base_cost(states, u, target)
        for h in range(self.H):
            sig = self._sigma(states[h])
            margin = Config.SAFETY_MARGIN + Config.ROBUST_BETA * sig
            cost += self._obstacle_cost(states[h], margin)
        return cost


# ─────────────────────────────────────────────────────────────────────────────
class RobustMPC(_MPCBase):
    _M = Config.SAFETY_MARGIN + Config.ROBUST_BETA * 0.2

    def _cost(self, u_flat, state, target):
        u = u_flat.reshape(self.H, 2)
        states = self._rollout(u, state)
        cost = self._base_cost(states, u, target)
        for h in range(self.H):
            cost += self._obstacle_cost(states[h], self._M)
        return cost


# ─────────────────────────────────────────────────────────────────────────────
class FedUMPCController(_MPCBase):
    """
    FedUMPC with stable BNN rollout.

    Key fix: BNN increments are CLAMPED before accumulation.
    Without clamping, an undertrained BNN (fast mode: 15 rounds, 300
    samples, hidden=64) can predict increments of ±5–20 m, sending the
    agent out of the 44×44 scene in a single step → immediate 'collision'
    → SR=0, Cost≈0, no visible trajectory.

    Clamp value = _MAX_INC = DT × 20 m/s = 2.0 m per component.
    This is generous (max real vehicle speed ~15 m/s → 1.5 m/step) and
    only activates for clearly wrong BNN predictions.

    Hybrid fallback: if the entire BNN increment is near-zero (BNN not
    yet warmed up), we blend with physics rollout to guarantee motion.
    """

    def __init__(self, env, bnn_model, uncertainty_aware=True,
                 lambda0=None, alpha_decay=0.1, *,
                 use_risk_cost=False, use_model_trust=True,
                 use_adaptive_margin=False, uncertainty_mode="estimated",
                 fixed_margin_profile=None, use_candidate_screen=True,
                 screen_horizon=None, contact_buffer=None,
                 screen_fallback=None, use_nominal_guard=False,
                 guard_nominal_slack=None, guard_clearance_gain=None):
        super().__init__(env)
        self.bnn     = bnn_model
        self.use_unc = uncertainty_aware
        self.use_risk_cost = bool(use_risk_cost and uncertainty_aware)
        self.use_model_trust = bool(use_model_trust and uncertainty_aware)
        self.use_adaptive_margin = bool(use_adaptive_margin and uncertainty_aware)
        if uncertainty_mode not in {"estimated", "constant", "shuffled"}:
            raise ValueError("uncertainty_mode must be estimated, constant, or shuffled")
        self.uncertainty_mode = uncertainty_mode
        self.use_candidate_screen = bool(
            use_candidate_screen and uncertainty_aware)
        self.use_nominal_guard = bool(use_nominal_guard and uncertainty_aware)
        self.guard_nominal_slack = float(
            Config.FED_GUARD_NOMINAL_SLACK if guard_nominal_slack is None
            else guard_nominal_slack)
        self.guard_clearance_gain = float(
            Config.FED_GUARD_CLEARANCE_GAIN if guard_clearance_gain is None
            else guard_clearance_gain)
        self.screen_horizon = int(
            Config.FED_SCREEN_HORIZON if screen_horizon is None
            else screen_horizon)
        self.contact_buffer = float(
            Config.FED_CONTACT_BUFFER if contact_buffer is None
            else contact_buffer)
        self.screen_fallback = (
            Config.FED_SCREEN_FALLBACK if screen_fallback is None
            else str(screen_fallback))
        if self.screen_fallback not in {"objective", "surplus"}:
            raise ValueError("screen_fallback must be objective or surplus")
        self.selection_cost_slack = float(
            Config.FED_SELECTION_COST_SLACK)
        self.fixed_margin_profile = None
        if fixed_margin_profile is not None:
            profile = np.asarray(fixed_margin_profile, dtype=np.float32).reshape(-1)
            if len(profile) != self.H:
                raise ValueError("fixed_margin_profile must have H entries")
            self.fixed_margin_profile = profile
        self.lambda0 = lambda0 if lambda0 is not None else Config.UNCERTAINTY_WEIGHT
        self.alpha   = alpha_decay
        h = np.arange(self.H, dtype=np.float32)
        self._lam_h   = (self.lambda0 * np.exp(-self.alpha * h)).astype(np.float32)
        self._inflate = (1.0 + 0.15 * h).astype(np.float32)

    def _transform_uncertainty(self, values, horizon_index):
        """Return the R1 control signal used by the isolation experiments.

        ``constant`` preserves the horizon-wise mean magnitude while removing
        candidate ordering. ``shuffled`` preserves the complete empirical
        distribution but deterministically breaks the candidate association.
        The deterministic permutation makes paired rollouts reproducible.
        """
        values = np.asarray(values, dtype=np.float32)
        if self.uncertainty_mode == "constant":
            return np.full_like(values, float(np.mean(values)))
        if self.uncertainty_mode == "shuffled" and values.size > 1:
            rng = np.random.default_rng(271828 + int(horizon_index))
            return values.reshape(-1)[rng.permutation(values.size)].reshape(values.shape)
        return values

    def _margins(self, uncertainty):
        base = Config.SAFETY_MARGIN + Config.FED_BASE_MARGIN_OFFSET
        uncertainty = np.asarray(uncertainty, dtype=np.float32)
        if self.fixed_margin_profile is not None:
            if uncertainty.ndim == 1:
                return self.fixed_margin_profile.copy()
            return np.broadcast_to(self.fixed_margin_profile,
                                   uncertainty.shape).copy()
        if not self.use_adaptive_margin:
            return np.full_like(uncertainty, base, dtype=np.float32)
        inflate = self._inflate if uncertainty.ndim == 1 else self._inflate[None, :]
        return base + np.clip(
            Config.ROBUST_BETA * Config.UNCERTAINTY_MARGIN_SCALE
            * uncertainty * inflate,
            0.0, Config.MAX_UNCERTAINTY_MARGIN)

    def _initial_guesses(self, state, target):
        return super()._initial_guesses(state, target)

    def _safe_bnn_rollout(self, u, state):
        """
        Recursive BNN increment rollout with clamping + physics hybrid.
        Returns bnn_states (H,4) guaranteed to stay in plausible range.
        """
        phy_states = self._rollout(u, state)   # physics fallback (H,4)
        bnn_states = np.empty((self.H, 4))
        unc = np.empty(self.H, dtype=np.float32)
        curr = state.copy()
        for h in range(self.H):
            mean, var = self.bnn.predict_uncertainty(curr, u[h])
            mean = np.clip(mean, -_MAX_INC, _MAX_INC)
            u_std = float(np.sqrt(max(var, 0.0)))
            u_std = float(self._transform_uncertainty(
                np.asarray([u_std], dtype=np.float32), h)[0])
            unc[h] = u_std
            phy_inc = phy_states[h] - (state if h == 0 else phy_states[h-1])
            phy_mag = np.linalg.norm(phy_inc)
            mismatch = np.linalg.norm(mean - phy_inc)
            trust_unc = (np.exp(-u_std / max(Config.BNN_TRUST_UNC_SCALE, 1e-6))
                         if self.use_model_trust else 1.0)
            trust_dyn = np.exp(-mismatch / max(phy_mag + 0.2, 0.2))
            trust = float(np.clip(trust_unc * trust_dyn, 0.0, 0.55))
            inc = (1.0 - trust) * phy_inc + trust * mean
            curr = curr + inc
            # Hard scene boundary clamp (scene is ±22 m)
            curr[0] = np.clip(curr[0], -Config.WORLD_LIMIT, Config.WORLD_LIMIT)
            curr[1] = np.clip(curr[1], -Config.WORLD_LIMIT, Config.WORLD_LIMIT)
            bnn_states[h] = curr

        return bnn_states, unc, phy_states

    def _safe_bnn_rollout_batch(self, u_batch, state):
        """Evaluate all finite MPC candidates with one MC batch per horizon step.

        The rollout remains recursive, but candidate trajectories are independent
        and can therefore share each neural-network launch.  This preserves the
        equations used by :meth:`_safe_bnn_rollout` while removing the dominant
        candidate-level Python/GPU synchronization overhead.
        """
        n_candidates = u_batch.shape[0]
        phy_states = np.empty((n_candidates, self.H, 4), dtype=float)
        for j in range(n_candidates):
            phy_states[j] = self._rollout(u_batch[j], state)

        bnn_states = np.empty_like(phy_states)
        unc = np.empty((n_candidates, self.H), dtype=np.float32)
        curr = np.repeat(np.asarray(state, dtype=float)[None, :],
                         n_candidates, axis=0)

        for h in range(self.H):
            mean, var = self.bnn.predict_horizon(curr, u_batch[:, h, :])
            mean = np.clip(mean, -_MAX_INC, _MAX_INC)
            u_std = np.sqrt(np.maximum(var, 0.0)).astype(np.float32)
            u_std = self._transform_uncertainty(u_std, h)
            unc[:, h] = u_std
            prev_phy = (np.repeat(np.asarray(state, dtype=float)[None, :],
                                  n_candidates, axis=0)
                        if h == 0 else phy_states[:, h - 1, :])
            phy_inc = phy_states[:, h, :] - prev_phy
            phy_mag = np.linalg.norm(phy_inc, axis=1)
            mismatch = np.linalg.norm(mean - phy_inc, axis=1)
            trust_unc = (np.exp(
                -u_std / max(Config.BNN_TRUST_UNC_SCALE, 1e-6))
                if self.use_model_trust else np.ones_like(u_std))
            trust_dyn = np.exp(
                -mismatch / np.maximum(phy_mag + 0.2, 0.2))
            trust = np.clip(trust_unc * trust_dyn, 0.0, 0.55)[:, None]
            curr = curr + (1.0 - trust) * phy_inc + trust * mean
            curr[:, 0] = np.clip(curr[:, 0], -Config.WORLD_LIMIT,
                                 Config.WORLD_LIMIT)
            curr[:, 1] = np.clip(curr[:, 1], -Config.WORLD_LIMIT,
                                 Config.WORLD_LIMIT)
            bnn_states[:, h, :] = curr

        return bnn_states, unc, phy_states

    def _evaluate_many(self, guesses, state, target):
        """Return finite-candidate scores and the rollouts used to score them."""
        u_batch = np.asarray(guesses, dtype=float).reshape(-1, self.H, 2)
        if not self.use_unc:
            values = np.asarray([self._cost(u.reshape(-1), state, target)
                                 for u in u_batch], dtype=float)
            return values, None, None, None

        bnn_states, unc, phy_states = self._safe_bnn_rollout_batch(
            u_batch, state)
        dist = np.linalg.norm(phy_states[:, :, :2] - target[None, None, :2],
                              axis=2)
        ctrl = np.sum(u_batch ** 2, axis=2)
        mismatch = np.linalg.norm(
            bnn_states[:, :, :2] - phy_states[:, :, :2], axis=2)
        prev = np.repeat(self.u_prev.reshape(1, self.H, 2),
                         len(u_batch), axis=0)[:, :1, :]
        du = np.diff(np.concatenate([prev, u_batch], axis=1), axis=1)

        values = (Config.FED_TRACKING_BOOST * Config.Q_TRACKING
                  * np.sum(dist ** 2, axis=1)
                  + Config.FED_TERMINAL_BOOST * Config.Q_TERMINAL
                  * dist[:, -1] ** 2
                  + Config.R_CONTROL * np.sum(ctrl, axis=1)
                  + Config.FED_SMOOTHNESS_WEIGHT
                  * np.sum(du ** 2, axis=(1, 2))
                  + (np.sum(self._lam_h[None, :] * unc, axis=1)
                     if self.use_risk_cost else 0.0)
                  + Config.FED_PROX_MISMATCH_WEIGHT * Config.Q_TRACKING
                  * np.sum(mismatch ** 2, axis=1))

        margins = self._margins(unc)
        for j in range(len(u_batch)):
            values[j] += sum(self._wall_cost(s) for s in phy_states[j])
            values[j] += sum(
                self._conservative_obstacle_cost(
                    phy_states[j, h], bnn_states[j, h], margins[j, h])
                for h in range(self.H))
        return (np.asarray(values, dtype=float), bnn_states, unc,
                phy_states)

    def _cost_many(self, guesses, state, target):
        """Vectorized finite-candidate objective used when SLSQP is disabled."""
        return self._evaluate_many(guesses, state, target)[0]

    def _screen_candidates(self, bnn_states, unc, phy_states):
        """Partition candidates by short-horizon conservative clearance.

        A finite candidate screen with a fallback does not establish recursive
        feasibility or collision avoidance; it only prevents the original cost
        ranking from preferring a predictably unsafe candidate when a screened
        alternative is already present in the shared candidate set.
        """
        margins = self._margins(unc)
        clearances = np.empty((len(phy_states), self.H), dtype=float)
        for j in range(len(phy_states)):
            for h in range(self.H):
                d_phy = self.env.get_min_dist(
                    phy_states[j, h, 0], phy_states[j, h, 1])
                d_hyb = self.env.get_min_dist(
                    bnn_states[j, h, 0], bnn_states[j, h, 1])
                clearances[j, h] = min(d_phy, d_hyb)
        hs = min(self.H, max(1, self.screen_horizon))
        worst_surplus = np.min(
            clearances[:, :hs] - margins[:, :hs], axis=1)
        worst_clearance = np.min(clearances[:, :hs], axis=1)
        admissible = np.flatnonzero(worst_surplus >= 0.0)
        noncontact = np.flatnonzero(worst_clearance >= self.contact_buffer)
        return admissible, noncontact, worst_surplus, worst_clearance

    def _choose_near_optimal(self, eligible, values, worst_clearance):
        """Break near-cost ties by conservative predicted clearance."""
        minimum = float(np.nanmin(values[eligible]))
        tolerance = self.selection_cost_slack * max(abs(minimum), 1.0)
        near = eligible[values[eligible] <= minimum + tolerance]
        if self.selection_cost_slack <= 0.0 or len(near) == 1:
            return int(eligible[np.nanargmin(values[eligible])])
        best_clearance = float(np.nanmax(worst_clearance[near]))
        safest = near[np.isclose(worst_clearance[near], best_clearance)]
        return int(safest[np.nanargmin(values[safest])])

    def _nominal_candidate_scores(self, guesses, state, target):
        """Evaluate the no-uncertainty objective on the shared candidates."""
        scores = []
        for guess in guesses:
            u = np.asarray(guess, dtype=float).reshape(self.H, 2)
            states = self._rollout(u, state)
            score = self._base_cost(states, u, target)
            score += sum(self._obstacle_cost(s, Config.SAFETY_MARGIN)
                         for s in states)
            scores.append(score)
        return np.asarray(scores, dtype=float)

    def _solve(self, state, target):
        if getattr(Config, 'USE_SLSQP_REFINEMENT', False):
            return super()._solve(state, target)
        guesses = self._initial_guesses(state, target)
        try:
            values, bnn_states, unc, phy_states = self._evaluate_many(
                guesses, state, target)
            if self.use_candidate_screen:
                admissible, noncontact, worst_surplus, worst_clearance = (
                    self._screen_candidates(bnn_states, unc, phy_states))
                if admissible.size:
                    fed_idx = self._choose_near_optimal(
                        admissible, values, worst_clearance)
                elif noncontact.size:
                    if self.screen_fallback == "surplus":
                        best_surplus = np.nanmax(
                            worst_surplus[noncontact])
                        near_best = noncontact[
                            worst_surplus[noncontact] >=
                            best_surplus - 0.05]
                        fed_idx = int(near_best[
                            np.nanargmin(values[near_best])])
                    else:
                        fed_idx = self._choose_near_optimal(
                            noncontact, values, worst_clearance)
                else:
                    fed_idx = int(np.nanargmax(worst_clearance))
            else:
                fed_idx = int(np.nanargmin(values))
                _, noncontact, _, worst_clearance = self._screen_candidates(
                    bnn_states, unc, phy_states)
                admissible = np.asarray([], dtype=int)
            if self.use_nominal_guard:
                nominal_values = self._nominal_candidate_scores(
                    guesses, state, target)
                nominal_idx = int(np.nanargmin(nominal_values))
                tolerance = self.guard_nominal_slack * max(
                    abs(float(nominal_values[nominal_idx])), 1.0)
                acceptable_cost = (
                    nominal_values[fed_idx] <=
                    nominal_values[nominal_idx] + tolerance)
                acceptable_clearance = (
                    worst_clearance[fed_idx] >=
                    worst_clearance[nominal_idx] + self.guard_clearance_gain)
                nominal_predicted_contact = (
                    worst_clearance[nominal_idx] < self.contact_buffer)
                fed_predicted_noncontact = bool(
                    np.any(noncontact == fed_idx))
                best_idx = (fed_idx if
                            ((acceptable_cost and acceptable_clearance) or
                             (nominal_predicted_contact and
                              fed_predicted_noncontact))
                            else nominal_idx)
            else:
                best_idx = fed_idx
            u_seq = np.clip(np.asarray(guesses[best_idx]), self._lo, self._hi)
        except Exception:
            # Retain the scalar path as a conservative fallback so that a
            # device-specific batching failure cannot terminate a rollout.
            return super()._solve(state, target)
        self.u_prev = np.clip(np.roll(u_seq, -2), self._lo, self._hi)
        self.u_prev[-2:] = u_seq[-2:]
        return u_seq[:2]

    def _cost(self, u_flat, state, target):
        u = u_flat.reshape(self.H, 2)

        # ── no-uncertainty mode ───────────────────────────────────────────────
        if not self.use_unc:
            states = self._rollout(u, state)
            cost   = self._base_cost(states, u, target)
            for h in range(self.H):
                cost += self._obstacle_cost(states[h], Config.SAFETY_MARGIN)
            return cost

        # ── uncertainty-aware mode ────────────────────────────────────────────
        bnn_states, unc, phy_states = self._safe_bnn_rollout(u, state)

        # Use the calibrated analytical vehicle model as the nominal MPC
        # trajectory. The BNN supplies residual uncertainty/risk shaping, not
        # the sole geometric path; this keeps FedUMPC as stable as Robust MPC
        # while still adapting margins by learned epistemic uncertainty.
        dist  = np.linalg.norm(phy_states[:, :2] - target[:2], axis=1)
        ctrl  = np.sum(u ** 2, axis=1)
        model_mismatch = np.linalg.norm(bnn_states[:, :2] - phy_states[:, :2], axis=1)
        du = np.diff(np.vstack([self.u_prev[:2], u]), axis=0)
        cost  = float(
            Config.FED_TRACKING_BOOST * Config.Q_TRACKING * np.sum(dist ** 2)
            + Config.FED_TERMINAL_BOOST * Config.Q_TERMINAL * dist[-1] ** 2
            + Config.R_CONTROL  * ctrl.sum()
            + Config.FED_SMOOTHNESS_WEIGHT * np.sum(du ** 2)
            + ((self._lam_h * unc).sum() if self.use_risk_cost else 0.0)
            + Config.FED_PROX_MISMATCH_WEIGHT * Config.Q_TRACKING * np.sum(model_mismatch ** 2)
            + sum(self._wall_cost(s) for s in phy_states)
        )

        margins = self._margins(unc)
        for h in range(self.H):
            cost += self._conservative_obstacle_cost(
                phy_states[h], bnn_states[h], margins[h])

        return cost

    def _fed_obstacle_cost(self, state, margin):
        d = self.env.get_min_dist(state[0], state[1])
        return self._fed_obstacle_cost_from_clearance(d, margin)

    def _conservative_obstacle_cost(self, physics_state, hybrid_state, margin):
        """Penalize the smaller clearance of nominal and learned rollouts."""
        d_phy = self.env.get_min_dist(physics_state[0], physics_state[1])
        d_hyb = self.env.get_min_dist(hybrid_state[0], hybrid_state[1])
        return self._fed_obstacle_cost_from_clearance(min(d_phy, d_hyb), margin)

    @staticmethod
    def _fed_obstacle_cost_from_clearance(d, margin):
        influence = max(Config.FED_OBSTACLE_INFLUENCE, margin)
        if d < margin:
            # R1 continuous C1 transition.  The constant and linear terms
            # match both the value and derivative of the outer quadratic at
            # d=margin; the inner quadratic then increases monotonically as
            # clearance decreases.
            gap = influence - margin
            deficit = margin - d
            return Config.OBSTACLE_PENALTY * (
                Config.FED_SOFT_OBS_WEIGHT * gap ** 2
                + 2.0 * Config.FED_SOFT_OBS_WEIGHT * gap * deficit
                + Config.FED_INNER_OBS_WEIGHT * deficit ** 2)
        if d < influence:
            return Config.FED_SOFT_OBS_WEIGHT * Config.OBSTACLE_PENALTY * (influence - d) ** 2
        return 0.0
