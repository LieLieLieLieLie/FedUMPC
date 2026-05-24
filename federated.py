# federated.py
"""
CHANGELOG v4 (2026-05-19):
──────────────────────────────────────────────────────────────────────────
FIX 1 – Scaffold v4强化修复 (v3仍爆炸: MSE=149619, Drift=1234):
  v3问题: 仅clamp delta(-10,10), 但c_local可累积超界; correction=
          c_server-c_local可能极大, 导致梯度校正爆炸→模型参数爆炸

  v4三层防护:
  (a) 梯度校正clamp: correction=clamp(c_server-c_local, -0.5, 0.5)
      → 防止单步梯度校正过大
  (b) c_local绝对值clamp: new_c=clamp(new_c, -2.0, 2.0)
      → 防止control variate累积爆炸
  (c) c_server更新clamp: 聚合后server variate也clamp(-1.0, 1.0)
      → 防止服务端传播放大误差

FIX 2 – Privacy accounting v2保留 (正确: 先累积RDP再转换):
  _compute_privacy_spent: total_rdp = n_steps * rdp_per_step (正确)

FIX 3 – DP-FedProx v2保留:
  mu_dp = max(Config.PROXIMAL_MU, 0.12)
  → DP噪声下更强近端正则化，防止过度漂移
──────────────────────────────────────────────────────────────────────────
"""

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from config import Config
from models import BNN


# ─── helpers ─────────────────────────────────────────────────────────────────
def _copy_weights(model):
    return {k: v.clone().detach() for k, v in model.state_dict().items()}


def _compute_drift(model, ref):
    return sum(
        (v.float() - ref[k].float()).norm(2).item()
        for k, v in model.state_dict().items()
    )


def _named_param_refs(model, state_dict):
    return {
        name: state_dict[name].to(Config.DEVICE)
        for name, _ in model.named_parameters()
        if name in state_dict
    }


def _dp_clip_and_noise(model, clip_norm, noise_mult, batch_size):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = math.sqrt(total_norm)
    scale = min(1.0, clip_norm / (total_norm + 1e-8))
    sensitivity = clip_norm * noise_mult / batch_size
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(scale)
            p.grad.data.add_(torch.randn_like(p.grad.data) * sensitivity)


def _compute_privacy_spent(noise_mult, clip_norm, n_data, batch_size,
                            n_steps, delta=1e-5):
    """
    v2修复: 先按n_steps累积RDP，再转换为(ε,δ)-DP。
    原代码: best_eps * n_steps (错误，导致所有ε目标映射到相同nm)
    正确:   total_rdp = n_steps * rdp_per_step，然后转换一次
    """
    if noise_mult <= 0:
        return float('inf')
    q = batch_size / max(n_data, 1)
    sigma = noise_mult
    best_eps = float('inf')
    for alpha in range(2, 65):
        rdp_per_step = alpha * q ** 2 / (2.0 * sigma ** 2)
        total_rdp    = n_steps * rdp_per_step   # 正确: 先累积
        eps = total_rdp + math.log(1.0 / delta) / (alpha - 1)
        if eps < best_eps:
            best_eps = eps
    return best_eps


# ─── base client ─────────────────────────────────────────────────────────────
class _BaseClient:
    def __init__(self, agent_id, mass=None):
        self.id          = agent_id
        self.mass        = mass or Config.MASS
        self.model       = BNN().to(Config.DEVICE)
        self.data_buffer = []
        self._init_w     = None

    def add_data(self, s, a, s_next):
        self.data_buffer.append((s, a, s_next - s))

    @property
    def num_samples(self):
        return len(self.data_buffer)

    def _prepare_tensors(self):
        s  = np.array([d[0] for d in self.data_buffer])
        a  = np.array([d[1] for d in self.data_buffer])
        ds = np.array([d[2] for d in self.data_buffer])
        xu = torch.FloatTensor(np.concatenate([s, a], axis=1)).to(Config.DEVICE)
        ds = torch.FloatTensor(ds).to(Config.DEVICE)
        return xu, ds

    def _test_mse(self, xu, ds):
        self.model.eval()
        with torch.no_grad():
            return nn.MSELoss()(self.model(xu), ds).item()

    def _iter_batches(self, n, xu, ds):
        idx = torch.randperm(n)
        for start in range(0, n, Config.BATCH_SIZE):
            bi = idx[start:start + Config.BATCH_SIZE]
            yield xu[bi], ds[bi]


# ─── 1. Local-only ───────────────────────────────────────────────────────────
class LocalClient(_BaseClient):
    def train_local(self):
        if self.num_samples < 10:
            return 0.0, self.model.state_dict(), 0.0
        xu, ds = self._prepare_tensors()
        mse0 = self._test_mse(xu, ds)
        self._init_w = _copy_weights(self.model)
        opt = optim.Adam(self.model.parameters(), lr=Config.LR)
        loss_fn = nn.MSELoss()
        self.model.train()
        n = self.num_samples
        for _ in range(Config.LOCAL_EPOCHS):
            for xub, dsb in self._iter_batches(n, xu, ds):
                opt.zero_grad()
                loss_fn(self.model(xub), dsb).backward()
                opt.step()
        mse1 = self._test_mse(xu, ds)
        return mse1, self.model.state_dict(), 0.0

    def train(self, global_weights, **kw):
        return self.train_local()


# ─── 2. FedAvg ───────────────────────────────────────────────────────────────
class FedAvgClient(_BaseClient):
    def train(self, global_weights, **kw):
        if self.num_samples < 10:
            return 0.0, self.model.state_dict(), 0.0
        self.model.load_state_dict(global_weights)
        self._init_w = _copy_weights(self.model)
        xu, ds = self._prepare_tensors()
        mse0 = self._test_mse(xu, ds)
        opt  = optim.SGD(self.model.parameters(), lr=Config.LR, momentum=0.9)
        loss_fn = nn.MSELoss()
        self.model.train()
        n = self.num_samples
        for _ in range(Config.LOCAL_EPOCHS):
            for xub, dsb in self._iter_batches(n, xu, ds):
                opt.zero_grad()
                loss_fn(self.model(xub), dsb).backward()
                opt.step()
        mse1 = self._test_mse(xu, ds)
        drift = _compute_drift(self.model, self._init_w)
        return mse1, self.model.state_dict(), drift


# ─── 3. Scaffold (v4强化修复) ─────────────────────────────────────────────────
class ScaffoldClient(_BaseClient):
    """
    Scaffold v4: 三层防护防止control variate爆炸。

    v3问题: MSE=149619, Drift=1234 — control variate累积爆炸
    v4修复:
      (a) correction = clamp(c_server - c_local, -0.5, 0.5) [每步梯度校正限幅]
      (b) new_c = clamp(new_c, -2.0, 2.0) [c_local绝对值上限]
      (c) NaN→0 guard保留
    """
    def __init__(self, agent_id, mass=None):
        super().__init__(agent_id, mass)
        self.c_local  = {k: torch.zeros_like(v)
                         for k, v in self.model.named_parameters()}
        self.c_server = {k: torch.zeros_like(v)
                         for k, v in self.model.named_parameters()}

    def train(self, global_weights, server_control=None, **kw):
        if server_control is not None:
            self.c_server = server_control
        if self.num_samples < 10:
            return 0.0, self.model.state_dict(), 0.0

        self.model.load_state_dict(global_weights)
        w0      = _copy_weights(self.model)
        self._init_w = w0
        xu, ds = self._prepare_tensors()
        mse0 = self._test_mse(xu, ds)

        opt     = optim.SGD(self.model.parameters(), lr=Config.LR)
        loss_fn = nn.MSELoss()
        self.model.train()
        n = self.num_samples

        for _ in range(Config.LOCAL_EPOCHS):
            for xub, dsb in self._iter_batches(n, xu, ds):
                opt.zero_grad()
                loss_fn(self.model(xub), dsb).backward()
                # (a) v4: clamp correction防止单步梯度校正过大
                for name, p in self.model.named_parameters():
                    if p.grad is not None:
                        raw_corr = (self.c_server[name].to(Config.DEVICE)
                                    - self.c_local[name].to(Config.DEVICE))
                        correction = torch.clamp(raw_corr, -0.5, 0.5)
                        p.grad.data.add_(-correction)
                opt.step()

        # 更新c_local
        w1 = _copy_weights(self.model)
        lr = Config.LR
        E  = Config.LOCAL_EPOCHS
        for k in self.c_local:
            delta = (w0[k].float() - w1[k].float()) / (E * lr + 1e-8)
            # v3: clamp delta
            delta = torch.clamp(delta, -10.0, 10.0)
            new_c = self.c_local[k].float() + delta - self.c_server[k].float()
            # (b) v4: c_local绝对值上限防止累积爆炸
            new_c = torch.clamp(new_c, -2.0, 2.0)
            # NaN guard保留
            new_c = torch.where(torch.isnan(new_c), torch.zeros_like(new_c), new_c)
            self.c_local[k] = new_c

        mse1 = self._test_mse(xu, ds)
        drift = _compute_drift(self.model, w0)
        return mse1, self.model.state_dict(), drift

    def get_delta_c(self, w0):
        return {k: self.c_local[k].clone() for k in self.c_local}


# ─── 4. MOON ─────────────────────────────────────────────────────────────────
class MOONClient(_BaseClient):
    def __init__(self, agent_id, mass=None):
        super().__init__(agent_id, mass)
        self.prev_model   = BNN().to(Config.DEVICE)
        self.global_model = BNN().to(Config.DEVICE)

    @staticmethod
    def _embed(model, xu):
        x = torch.relu(model.fc1(xu))
        x = model.dropout(x)
        x = torch.relu(model.fc2(x))
        return x

    @staticmethod
    def _cos_sim(a, b):
        a = nn.functional.normalize(a, dim=-1)
        b = nn.functional.normalize(b, dim=-1)
        return (a * b).sum(-1)

    def train(self, global_weights, **kw):
        if self.num_samples < 10:
            return 0.0, self.model.state_dict(), 0.0
        self.global_model.load_state_dict(global_weights)
        self.model.load_state_dict(global_weights)
        self._init_w = _copy_weights(self.model)
        xu, ds = self._prepare_tensors()
        mse0   = self._test_mse(xu, ds)
        opt     = optim.Adam(self.model.parameters(), lr=Config.LR)
        loss_fn = nn.MSELoss()
        mu = Config.MOON_MU; tau = Config.MOON_TEMP
        self.model.train()
        self.global_model.eval(); self.prev_model.eval()
        n = self.num_samples
        for _ in range(Config.LOCAL_EPOCHS):
            for xub, dsb in self._iter_batches(n, xu, ds):
                opt.zero_grad()
                pred  = self.model(xub)
                mse   = loss_fn(pred, dsb)
                z   = self._embed(self.model,        xub)
                z_g = self._embed(self.global_model, xub).detach()
                z_p = self._embed(self.prev_model,   xub).detach()
                pos  = (self._cos_sim(z, z_g) / tau).exp()
                neg  = (self._cos_sim(z, z_p) / tau).exp()
                l_con = -torch.log(pos / (pos + neg + 1e-8)).mean()
                (mse + mu * l_con).backward()
                opt.step()
        self.prev_model.load_state_dict(_copy_weights(self.model))
        mse1 = self._test_mse(xu, ds)
        drift = _compute_drift(self.model, self._init_w)
        return mse1, self.model.state_dict(), drift


# ─── 5. FedProx ──────────────────────────────────────────────────────────────
class FedProxClient(_BaseClient):
    def train(self, global_weights, **kw):
        if self.num_samples < 10:
            return 0.0, self.model.state_dict(), 0.0
        self.model.load_state_dict(global_weights)
        self._init_w = _copy_weights(self.model)
        xu, ds = self._prepare_tensors()
        mse0 = self._test_mse(xu, ds)
        opt  = optim.Adam(self.model.parameters(),
                          lr=Config.LR * Config.FEDPROX_LR_MULT)
        loss_fn = nn.MSELoss()
        self.model.train()
        n = self.num_samples
        for _ in range(Config.LOCAL_EPOCHS):
            wg_params = _named_param_refs(self.model, global_weights)
            for xub, dsb in self._iter_batches(n, xu, ds):
                opt.zero_grad()
                mse  = loss_fn(self.model(xub), dsb)
                prox = sum(
                    (p - wg_params[name]).norm(2) ** 2
                    for name, p in self.model.named_parameters()
                )
                (mse + (Config.PROXIMAL_MU / 2.0) * prox).backward()
                opt.step()
        mse1 = self._test_mse(xu, ds)
        drift = _compute_drift(self.model, self._init_w)
        return mse1, self.model.state_dict(), drift


# ─── 6. DP-FedAvg ────────────────────────────────────────────────────────────
class DPFedAvgClient(FedAvgClient):
    def __init__(self, agent_id, mass=None, noise_mult=None, clip_norm=None):
        super().__init__(agent_id, mass)
        self.noise_mult = noise_mult or Config.DP_NOISE_MULT
        self.clip_norm  = clip_norm  or Config.DP_MAX_GRAD_NORM

    def train(self, global_weights, **kw):
        if self.num_samples < 10:
            return 0.0, self.model.state_dict(), 0.0
        self.model.load_state_dict(global_weights)
        self._init_w = _copy_weights(self.model)
        xu, ds = self._prepare_tensors()
        mse0 = self._test_mse(xu, ds)
        opt  = optim.SGD(self.model.parameters(), lr=Config.LR, momentum=0.9)
        loss_fn = nn.MSELoss()
        self.model.train()
        n = self.num_samples
        for _ in range(Config.LOCAL_EPOCHS):
            for xub, dsb in self._iter_batches(n, xu, ds):
                opt.zero_grad()
                loss_fn(self.model(xub), dsb).backward()
                _dp_clip_and_noise(self.model, self.clip_norm,
                                   self.noise_mult, len(xub))
                opt.step()
        mse1 = self._test_mse(xu, ds)
        drift = _compute_drift(self.model, self._init_w)
        return mse1, self.model.state_dict(), drift

    def privacy_spent(self):
        n_steps = Config.LOCAL_EPOCHS * math.ceil(
            self.num_samples / Config.BATCH_SIZE)
        return _compute_privacy_spent(
            self.noise_mult, self.clip_norm,
            self.num_samples, Config.BATCH_SIZE,
            n_steps, Config.DP_DELTA)


# ─── 7. DP-FedProx (v5: SGD替换Adam，降低DP噪声放大效应) ───────────────────
class DPFedProxClient(FedProxClient):
    """
    FedProx + DP-SGD.

    v5改进: 改用SGD+momentum替代Adam。
    原因: Adam的自适应学习率会放大DP Gaussian噪声的方差，导致drift=2.1
    (DP-FedAvg用SGD只有drift=0.36)。改用SGD后drift预计降至~0.5。
    mu_dp=0.12保留(DP噪声下需要更强近端正则化)。
    """
    def __init__(self, agent_id, mass=None, noise_mult=None, clip_norm=None):
        super().__init__(agent_id, mass)
        self.noise_mult = noise_mult or Config.DP_NOISE_MULT
        self.clip_norm  = clip_norm  or Config.DP_MAX_GRAD_NORM

    def train(self, global_weights, **kw):
        if self.num_samples < 10:
            return 0.0, self.model.state_dict(), 0.0
        self.model.load_state_dict(global_weights)
        self._init_w = _copy_weights(self.model)
        xu, ds = self._prepare_tensors()
        mse0 = self._test_mse(xu, ds)
        # v5: SGD+momentum (与DP-FedAvg对齐), 避免Adam放大DP噪声→高drift
        opt  = optim.SGD(self.model.parameters(), lr=Config.LR, momentum=0.9)
        loss_fn = nn.MSELoss()
        self.model.train()
        n = self.num_samples
        mu_dp = max(Config.PROXIMAL_MU, 0.12)   # 更强近端正则化
        for _ in range(Config.LOCAL_EPOCHS):
            wg_params = _named_param_refs(self.model, global_weights)
            for xub, dsb in self._iter_batches(n, xu, ds):
                opt.zero_grad()
                mse  = loss_fn(self.model(xub), dsb)
                prox = sum(
                    (p - wg_params[name]).norm(2) ** 2
                    for name, p in self.model.named_parameters()
                )
                (mse + (mu_dp / 2.0) * prox).backward()
                _dp_clip_and_noise(self.model, self.clip_norm,
                                   self.noise_mult, len(xub))
                opt.step()
        mse1 = self._test_mse(xu, ds)
        drift = _compute_drift(self.model, self._init_w)
        return mse1, self.model.state_dict(), drift

    def privacy_spent(self):
        n_steps = Config.LOCAL_EPOCHS * math.ceil(
            self.num_samples / Config.BATCH_SIZE)
        return _compute_privacy_spent(
            self.noise_mult, self.clip_norm,
            self.num_samples, Config.BATCH_SIZE,
            n_steps, Config.DP_DELTA)


# ─── Server ───────────────────────────────────────────────────────────────────
class FedServer:
    def __init__(self):
        self.global_model = BNN().to(Config.DEVICE)
        self.c_server = {k: torch.zeros_like(v)
                         for k, v in self.global_model.named_parameters()}

    def aggregate(self, clients):
        total = sum(c.num_samples for c in clients)
        if total == 0:
            return self.global_model.state_dict()
        weights = [c.num_samples / total for c in clients]
        new_dict = {k: torch.zeros_like(v)
                    for k, v in self.global_model.state_dict().items()}
        for wi, c in zip(weights, clients):
            for k, v in c.model.state_dict().items():
                new_dict[k] += wi * v.float()
        self.global_model.load_state_dict(new_dict)
        return new_dict

    def aggregate_scaffold(self, clients, w0_dict):
        new_dict = self.aggregate(clients)
        n = len(clients)
        for k in self.c_server:
            delta = sum(
                c.get_delta_c(w0_dict)[k].float()
                for c in clients if isinstance(c, ScaffoldClient)
            ) / max(n, 1)
            delta = torch.where(torch.isnan(delta), torch.zeros_like(delta), delta)
            new_c = self.c_server[k].float() + delta
            # (c) v4: server variate也clamp，防止传播放大
            new_c = torch.clamp(new_c, -1.0, 1.0)
            self.c_server[k] = new_c
        return new_dict

    def client_drift(self, clients):
        wg = self.global_model.state_dict()
        return sum(
            sum((wl.float() - wg[k].float()).norm(2).item()
                for k, wl in c.model.state_dict().items())
            for c in clients
        ) / max(len(clients), 1)
