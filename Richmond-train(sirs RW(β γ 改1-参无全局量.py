# -*- coding: utf-8 -*-

"""

Richmond Park – GNN+RNN SIR（VI版，β=RNN均值 + 变分随机残差）:

- γ(>0)、(可选)ω(>0)、σ_e(>0) 仍用 LogNormal VI

- β：logβ_t = logβ_det_t (由 GNN+GRU 给出) + u_t

      q(u) = N(μ_u, diag(σ_u^2))  （重参数化采样）

      先验:  u_0 ~ N(prior_u0_mu, prior_u0_sigma^2),

             u_t - u_{t-1} ~ N(0, prior_u_rw_sigma^2)

- 训练目标: 最大化 ELBO = E_q[loglik + log p(θ,u) - log q(θ,u)] + 单调性软约束

- 导出：R/S/I/β 的后验带、γ/ω样本、走动验证

"""

'''

改动1：免疫缺失率调整更接近生物规律的先验

改动2：收紧β自由度 回溯更多给w

'''



import argparse, os, math, sys, pathlib

from dataclasses import dataclass

from typing import Dict, Tuple, Optional



import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F

import matplotlib.pyplot as plt



# -- Add src/ to import path for LaGNA-style utilities -------------------------

_REPO_ROOT = pathlib.Path(__file__).resolve().parent

_SRC_DIR = _REPO_ROOT / "src"

if _SRC_DIR.exists():

    src_str = str(_SRC_DIR)

    if src_str not in sys.path:

        sys.path.insert(0, src_str)



from lagna_sirs import (

    build_transition_feature_matrix,

    export_symbolic_summary,

    fit_symbolic_model,

)

from lagna_sirs.symbolic_regression import SymbolicModel

from tree_sirs_sim import GraphSIRSMessagePassing



# ============== 可选：PyG ==============

try:

    from torch_geometric.data import Data as GeometricData

    from torch_geometric.nn import GCNConv

    _HAS_PYG = True

except Exception:

    GeometricData = None
    _HAS_PYG = False



# ============== 实用函数 ==============

LOG2PI = math.log(2.0 * math.pi)



def quantile_stats(stack: np.ndarray, axis: int = 0):

    mean = stack.mean(axis=axis)

    q25, q75 = np.percentile(stack, [25, 75], axis=axis)

    q2p5, q97p5 = np.percentile(stack, [2.5, 97.5], axis=axis)

    return {"mean": mean, "q25": q25, "q75": q75, "q2p5": q2p5, "q97p5": q97p5}


def extract_graph_payload(gdata):
    if _HAS_PYG and isinstance(gdata, GeometricData):
        edge_index = gdata.edge_index
        edge_attr = getattr(gdata, "edge_distance", None)
        if edge_attr is None and hasattr(gdata, "edge_attr"):
            edge_attr = gdata.edge_attr
        if edge_attr is not None and edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)
        coords = getattr(gdata, "coords", None)
        first_year = getattr(gdata, "first_year", None)
        num_nodes = gdata.num_nodes
    else:
        edge_index = gdata["edge_index"]
        edge_attr = gdata.get("edge_distance") or gdata.get("edge_attr")
        if edge_attr is not None and edge_attr.dim() > 1:
            edge_attr = edge_attr.squeeze(-1)
        coords = gdata.get("coords")
        first_year = gdata.get("first_year")
        num_nodes = gdata["x"].size(0)
    return {
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "coords": coords,
        "first_year": first_year,
        "num_nodes": num_nodes,
    }


def build_initial_node_states(first_year: torch.Tensor, start_year: int, device: torch.device):
    if first_year is None:
        raise ValueError("Graph data must include first_year tensor for initialization.")
    fy = first_year.to(device=device, dtype=torch.long)
    recovered_mask = (fy <= int(start_year)).to(device=device, dtype=torch.float32)
    S0 = 1.0 - recovered_mask
    I0 = torch.zeros_like(S0)
    R0 = recovered_mask.clone()
    C0 = R0.clone()
    return S0, I0, R0, C0



def load_graph_and_series(data_dir: str, no_pyg: bool = False):

    import os, numpy as np, pandas as pd

    # 1) 读图数据

    nodes_path = os.path.join(data_dir, "nodes.csv")

    edge_path  = os.path.join(data_dir, "edge_index.npy")

    ys_path    = os.path.join(data_dir, "yearly_series.csv")



    nodes = pd.read_csv(nodes_path)

    ys    = pd.read_csv(ys_path)



    X_cols = [c for c in nodes.columns if c.lower() in ['x', 'easting']]

    Y_cols = [c for c in nodes.columns if c.lower() in ['y', 'northing']]

    if X_cols and Y_cols:

        coords_raw = nodes[[X_cols[0], Y_cols[0]]].values.astype('float32')

    else:

        coords_raw = np.zeros((len(nodes), 2), dtype='float32')

    X = (coords_raw - coords_raw.mean(0, keepdims=True)) / (coords_raw.std(0, keepdims=True) + 1e-6)



    first_year_col = None

    for cand in ['first_year', 'firstyear', 'year_first']:

        if cand in nodes.columns:

            first_year_col = cand

            break

    first_year = nodes[first_year_col].to_numpy(dtype='int32') if first_year_col else np.full(len(nodes), -1, dtype='int32')



    E = np.load(edge_path)  # shape [2, E]

    if coords_raw.size > 0:

        diff = coords_raw[E[0]] - coords_raw[E[1]]

        edge_dist = np.linalg.norm(diff, axis=1).astype('float32')

    else:

        edge_dist = np.ones(E.shape[1], dtype='float32')



    edge_index_t = torch.tensor(E, dtype=torch.long)

    edge_attr_t = torch.tensor(edge_dist, dtype=torch.float32)

    x_t = torch.tensor(X, dtype=torch.float32)

    coords_t = torch.tensor(coords_raw, dtype=torch.float32)

    first_year_t = torch.tensor(first_year, dtype=torch.long)



    # PyG or fallback

    if not no_pyg:

        try:

            from torch_geometric.data import Data

            gdata = Data(x=x_t, edge_index=edge_index_t, edge_attr=edge_attr_t.unsqueeze(-1))

            gdata.coords = coords_t

            gdata.first_year = first_year_t

            gdata.edge_distance = edge_attr_t

        except Exception:

            gdata = {'x': x_t, 'edge_index': edge_index_t, 'edge_attr': edge_attr_t, 'coords': coords_t, 'first_year': first_year_t}

    else:

        gdata = {'x': x_t, 'edge_index': edge_index_t, 'edge_attr': edge_attr_t, 'coords': coords_t, 'first_year': first_year_t}



    # 2) 读时间序列

    # 统一列名小写视图

    lowcols = [c.lower() for c in ys.columns]

    def pick_col(name_candidates):

        for k in name_candidates:

            if k in lowcols:

                return ys.columns[lowcols.index(k)]

        return None



    # 年份

    ycol = pick_col(["year"])

    if ycol is None:

        raise ValueError("yearly_series.csv 必须包含 Year 列")

    years = ys[ycol].to_numpy().astype("int32")



    # ——优先选择累计观测列（SIRS 模式对齐 C）——

    # 1. C_events_allrows  2. C_events_treeyear  3. 任何包含 'cum' 的列  4. R

    R_col = pick_col(["c_events_allrows"]) or pick_col(["c_events_treeyear"])

    if R_col is None:

        for c in ys.columns:

            if ("cum" in c.lower()):

                R_col = c; break

    if R_col is None and "R" in ys.columns:

        R_col = "R"

    if R_col is None:

        raise ValueError("找不到累计观测列，请提供 C_events_allrows 或 C_events_treeyear（或含 'cum' 的列/列名 R）")



    R_obs = ys[R_col].to_numpy(dtype="float32")



    # R_unique（可选）

    Runiq_obs = ys["R_unique"].to_numpy(dtype="float32") if "R_unique" in ys.columns else None



    # 年度新增量（仅作为时间特征滞后输入使用，缺则由累计差分）

    dR = np.diff(R_obs, prepend=R_obs[0]).astype("float32")



    # nests（可选，不强制）

    nests_col = None

    for c in ys.columns:

        if c.lower() in ["nests_removed", "nests", "events_allrows", "events_treeyear"]:

            nests_col = c; break

    nests = ys[nests_col].to_numpy(dtype="float32") if nests_col else dR



    return gdata, years, R_obs, Runiq_obs, dR, nests, X.shape[0]





# ============== 模型 ==============

class GraphEncoder(nn.Module):

    def __init__(self, in_dim: int, hidden: int, use_pyg: bool = True):

        super().__init__()

        self.use_pyg = use_pyg and _HAS_PYG

        if self.use_pyg:

            self.conv1 = GCNConv(in_dim, hidden)

            self.conv2 = GCNConv(hidden, hidden)

        else:

            self.lin1 = nn.Linear(in_dim, hidden)

            self.lin2 = nn.Linear(hidden, hidden)



    def forward(self, gdata):

        if self.use_pyg:

            x = torch.relu(self.conv1(gdata.x, gdata.edge_index))

            x = torch.relu(self.conv2(x, gdata.edge_index))

        else:

            x = torch.relu(self.lin1(gdata['x']))

            x = torch.relu(self.lin2(x))

        return x.mean(0, keepdim=True)  # 图级 embedding



class BetaRNN(nn.Module):

    def __init__(self, g_dim: int, hidden: int = 64):

        super().__init__()

        self.gru = nn.GRU(input_size=3 + g_dim, hidden_size=hidden, batch_first=True)

        self.head = nn.Linear(hidden, 1)



    def forward(self, seq_inputs: torch.Tensor, g: torch.Tensor):

        T = seq_inputs.size(1)

        g_rep = g.unsqueeze(1).expand(-1, T, -1)

        z = torch.cat([seq_inputs, g_rep], dim=-1)

        h, _ = self.gru(z)

        beta = F.softplus(self.head(h)).squeeze(-1).squeeze(0)  # [T] >= 0

        return beta  # 标量年尺度 β_det 序列



class GammaHead(nn.Module):

    """A small GRU head to produce gamma_det(t) > 0, similar to BetaRNN but lighter."""

    def __init__(self, g_dim: int, hidden: int = 32):

        super().__init__()

        self.gru = nn.GRU(input_size=3 + g_dim, hidden_size=hidden, batch_first=True)

        self.head = nn.Linear(hidden, 1)

    def forward(self, seq_inputs: torch.Tensor, g: torch.Tensor):

        T = seq_inputs.size(1)

        g_rep = g.unsqueeze(1).expand(-1, T, -1)

        z = torch.cat([seq_inputs, g_rep], dim=-1)

        h, _ = self.gru(z)

        gamma_det = F.softplus(self.head(h)).squeeze(-1).squeeze(0) + 1e-6

        return gamma_det



class OmegaHead(nn.Module):

    """A small GRU head to produce omega_det(t) > 0, symmetric to GammaHead."""

    def __init__(self, g_dim: int, hidden: int = 32):

        super().__init__()

        self.gru = nn.GRU(input_size=3 + g_dim, hidden_size=hidden, batch_first=True)

        self.head = nn.Linear(hidden, 1)

    def forward(self, seq_inputs: torch.Tensor, g: torch.Tensor):

        T = seq_inputs.size(1)

        g_rep = g.unsqueeze(1).expand(-1, T, -1)

        z = torch.cat([seq_inputs, g_rep], dim=-1)

        h, _ = self.gru(z)

        omega_det = F.softplus(self.head(h)).squeeze(-1).squeeze(0) + 1e-6

        return omega_det

class GNN_SIR_Core(nn.Module):

    """GNN encoder + temporal RNN + graph SIRS forward module."""

    def __init__(self, N: float, node_feat_dim: int, hidden: int = 64, use_pyg: bool = True,
                 monthly_steps: int = 12, sirs: bool = False):
        super().__init__()
        self.N = float(N)
        self.enc = GraphEncoder(node_feat_dim, hidden, use_pyg=use_pyg)
        self.temporal = BetaRNN(g_dim=hidden, hidden=hidden)
        self.gamma_head = GammaHead(g_dim=hidden, hidden=max(16, hidden // 2))
        self.omega_head = OmegaHead(g_dim=hidden, hidden=max(16, hidden // 2))
        self.monthly_steps = monthly_steps
        self.sirs = sirs
        self.graph_module = None

    def set_graph(self, edge_index: torch.Tensor, edge_attr: torch.Tensor, num_nodes: int,
                  node_weights: torch.Tensor = None):
        edge_index = edge_index.detach()
        edge_attr = edge_attr.detach() if edge_attr is not None else None
        node_weights = node_weights.detach() if node_weights is not None else None
        self.graph_module = GraphSIRSMessagePassing(
            edge_index=edge_index,
            edge_distance=edge_attr,
            num_nodes=int(num_nodes),
            monthly_steps=self.monthly_steps,
            sirs=self.sirs,
            node_weights=node_weights,
        ).to(edge_index.device)

    def run_simulation(self, beta_seq: torch.Tensor, gamma_seq: torch.Tensor,
                       omega_seq: torch.Tensor, initial_state):
        if self.graph_module is None:
            raise RuntimeError("Graph simulator not initialized. Call set_graph() first.")
        return self.graph_module(beta_seq, gamma_seq, omega_seq, initial_state)


class LogNormalVI(nn.Module):

    """q(x)=LogNormal(μ,σ)，重参数化: logx=μ+σ·ε, x=exp(logx)"""

    def __init__(self, mu_init: float, rho_init: float = -0.5):

        super().__init__()

        self.mu = nn.Parameter(torch.tensor(float(mu_init)))

        self.rho = nn.Parameter(torch.tensor(float(rho_init)))  # σ = softplus(rho)



    def sample(self, n: int, device):

        eps = torch.randn(n, device=device)

        sigma = F.softplus(self.rho) + 1e-6

        logx = self.mu + sigma * eps

        x = torch.exp(logx)

        return x, logx, sigma



    def log_q(self, x: torch.Tensor) -> torch.Tensor:

        sigma = F.softplus(self.rho) + 1e-6

        return -0.5 * (((torch.log(x) - self.mu) / sigma) ** 2 + LOG2PI) - torch.log(x) - torch.log(sigma)



def log_lognormal_pdf(x: torch.Tensor, mu: float, sigma: float) -> torch.Tensor:

    sigma_t = torch.tensor(float(sigma), dtype=torch.float32, device=x.device)

    mu_t = torch.tensor(float(mu), dtype=torch.float32, device=x.device)

    return -0.5 * (((torch.log(x) - mu_t) / sigma_t) ** 2 + LOG2PI) - torch.log(x) - torch.log(sigma_t)



# NEW(u-VI) —— 向量高斯 VI（对 u_{1:T-1} 做对角协方差的近似）

class GaussianDiagVI(nn.Module):

    def __init__(self, dim: int, mu_init: float = 0.0, rho_init: float = -0.3):

        super().__init__()

        self.mu = nn.Parameter(torch.full((dim,), float(mu_init)))

        self.rho = nn.Parameter(torch.full((dim,), float(rho_init)))  # σ = softplus(rho)



    def sample(self, n: int, device):

        # 返回 samples:[n, dim]，以及 σ:[dim]

        eps = torch.randn(n, self.mu.numel(), device=device)

        sigma = F.softplus(self.rho) + 1e-6

        samples = self.mu + eps * sigma

        return samples, sigma



    def log_q(self, x: torch.Tensor) -> torch.Tensor:

        # x:[n,dim] or [dim]; 返回标量（按维度求平均，避免尺度爆炸）

        if x.dim() == 1: x = x.unsqueeze(0)

        sigma = F.softplus(self.rho) + 1e-6  # [dim]

        z = (x - self.mu) / sigma  # [n,dim]

        logp = -0.5 * (z**2 + LOG2PI) - torch.log(sigma)

        return logp.mean()  # 平均到每维/每样本



# u 的随机游走先验的 log 密度

def log_prior_u_rw(u: torch.Tensor, mu0: float, sigma0: float, tau: float) -> torch.Tensor:

    # u:[dim]；返回标量（同样按维度平均）

    dim = u.numel()

    u0 = u[0]

    deltas = u[1:] - u[:-1]

    logp0 = -0.5 * (((u0 - mu0) / sigma0) ** 2 + LOG2PI) - math.log(max(sigma0, 1e-12))

    logprw = -0.5 * ((deltas / tau) ** 2 + LOG2PI) - math.log(max(tau, 1e-12))

    return (logp0 + logprw.sum()) / float(dim)



# ============== 配置 ==============

@dataclass

class TrainCfg:

    # 训练

    epochs: int = 2500

    lr: float = 3e-3

    weight_decay: float = 1e-5

    hidden: int = 64

    monthly_steps: int = 12

    use_pyg: bool = True

    sirs: bool = False



    # 总体规模 & 初值

    N_base: float = 40000.0

    allow_N_perturb: bool = False

    N_low_ratio: float = 0.9

    N_high_ratio: float = 1.1

    I0_scale_low: float = 0.8

    I0_scale_high: float = 1.5



    # 观测模型

    heteroskedastic_obs: bool = True

    kappa_obs: float = 50.0



    # γ/σ_e/ω 的先验（对数空间）

    prior_gamma_mu: float = float(np.log(1.0))

    prior_gamma_sigma: float = 0.7

    prior_sigmae_mu: float = float(np.log(0.5))

    prior_sigmae_sigma: float = 0.6

    prior_omega_mu: float = float(np.log(0.5)) #修改免疫衰减的先验

    prior_omega_sigma: float = 0.7



    # VI 采样

    vi_mc_samples: int = 4

    vi_draws_for_paths: int = 200



    # NEW(u-VI) —— u 的先验（初值+随机游走）

    prior_u0_mu: float = 0.0

    prior_u0_sigma: float = 0.5

    prior_u_rw_sigma: float = 0.25  # τ_u



    # 单调性约束权重

    monotonic_weight: float = 1.0



    # 走动验证

    do_walk_forward: bool = True

    min_train_years: int = 3

    max_train_years: Optional[int] = None

    out_dir: str = "vi_out"



# ============== 初始化 & 序列构造 ==============

def sample_init(R_obs: np.ndarray, dR: np.ndarray, cfg: TrainCfg) -> Tuple[float, float, float, float]:

    R0 = float(R_obs[0])

    dR1 = float(max(dR[1] if len(dR) > 1 else (R_obs[0] if R_obs[0] > 0 else 1.0), 1.0))

    N = (np.random.uniform(cfg.N_low_ratio, cfg.N_high_ratio) * cfg.N_base) if cfg.allow_N_perturb else cfg.N_base

    I0 = np.random.uniform(cfg.I0_scale_low, cfg.I0_scale_high) * dR1

    S0 = N - R0 - I0

    if S0 <= 0: return sample_init(R_obs, dR, cfg)

    return float(N), float(S0), float(I0), float(R0)



def build_seq_inputs(dR: np.ndarray, nests: np.ndarray, T_use: Optional[int], device):

    if T_use is None: T_use = len(dR)

    dR_norm = (dR[:T_use] / max(np.max(dR[:T_use]), 1.0)).astype(np.float32)

    nests_norm = (nests[:T_use] / max(np.max(nests[:T_use]), 1.0)).astype(np.float32)

    ones = np.ones_like(dR_norm, dtype=np.float32)

    seq = np.stack([np.roll(dR_norm,1), np.roll(nests_norm,1), ones], axis=-1)[1:]  # [T-1,3]

    return torch.tensor(seq, dtype=torch.float32, device=device).unsqueeze(0)  # [1,T-1,3]



# ============== 训练（VI） ==============

def train_vi_once(

    # gdata, years, R_obs, dR, nests, cfg: TrainCfg,

    gdata, years, R_obs, Runiq_obs, dR, nests, cfg: TrainCfg,

    train_T=None, verbose=True, dev="cuda" if torch.cuda.is_available() else "cpu"

):



    if train_T is None:

        train_T = len(years)

    # 观测张量

    R_true = torch.tensor(R_obs[:train_T], dtype=torch.float32, device=dev)

    Ru_true = (torch.tensor(Runiq_obs[:train_T], dtype=torch.float32, device=dev)

               if (Runiq_obs is not None) else None)

    # graph -> device

    if _HAS_PYG and isinstance(gdata, GeometricData):

        gdev = gdata.to(dev); node_feat_dim = gdev.x.size(1)

    else:

        gdev = {k: v.to(dev) for k, v in gdata.items()}; node_feat_dim = gdev['x'].size(1)



    T_full = len(R_obs)

    train_T = int(T_full if train_T is None else train_T)

    assert 2 <= train_T <= T_full, "train_T 越界"



    N, S0_f, I0_f, R0_f = sample_init(R_obs, dR, cfg)

    S0 = torch.tensor(S0_f, dtype=torch.float32, device=dev)

    I0 = torch.tensor(I0_f, dtype=torch.float32, device=dev)

    R0 = torch.tensor(R0_f, dtype=torch.float32, device=dev)



    # 核心确定性模块

    core = GNN_SIR_Core(N=N, node_feat_dim=node_feat_dim, hidden=cfg.hidden,

                        use_pyg=cfg.use_pyg and _HAS_PYG,

                        monthly_steps=cfg.monthly_steps, sirs=cfg.sirs).to(dev)

    graph_payload = extract_graph_payload(gdev)
    edge_index_sim = graph_payload["edge_index"].to(dev)
    edge_attr_sim = graph_payload["edge_attr"]
    if edge_attr_sim is None:
        edge_attr_sim = torch.ones(edge_index_sim.size(1), device=dev, dtype=torch.float32)
    else:
        edge_attr_sim = edge_attr_sim.to(device=dev, dtype=torch.float32)
        if edge_attr_sim.dim() > 1:
            edge_attr_sim = edge_attr_sim.squeeze(-1)
    core.set_graph(edge_index=edge_index_sim, edge_attr=edge_attr_sim, num_nodes=graph_payload["num_nodes"])
    node_state0 = build_initial_node_states(graph_payload["first_year"], int(years[0]), dev)



    # 变分参数：γ、σ_e、(ω) + NEW(u)

    # 仅保留 σ_e、β的u、γ的u、ω的u（γ/ω不再有“全局标量”）

    q_sigmae = LogNormalVI(mu_init=float(cfg.prior_sigmae_mu), rho_init=-0.5).to(dev)

    q_u = GaussianDiagVI(dim=train_T - 1, mu_init=0.0, rho_init=-0.3).to(dev)  # β 的 u

    q_u_gamma = GaussianDiagVI(dim=train_T - 1, mu_init=0.0, rho_init=-1.2).to(dev)  # γ 的 uγ

    q_u_omega = GaussianDiagVI(dim=train_T - 1, mu_init=0.0, rho_init=-1.2).to(dev)  # ω 的 uω



    params = (list(core.parameters())

              + list(q_sigmae.parameters())

              + list(q_u.parameters()) + list(q_u_gamma.parameters()) + list(q_u_omega.parameters()))

    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)



    # 数据

    R_true = torch.tensor(R_obs[:train_T], dtype=torch.float32, device=dev)

    seq_inputs = build_seq_inputs(dR, nests, train_T, dev)  # [1,train_T-1,3]



    best = {"loss": float('inf'), "state_core": None, "q_snap": None}



    for ep in range(1, cfg.epochs + 1):

        core.train(); opt.zero_grad()

        K = cfg.vi_mc_samples

        elbo_acc = 0.0

        penalty_acc = 0.0



        for _ in range(K):

            # 采样 θ = {γ, σ_e, (ω)}:



            sigmae_s, _, _ = q_sigmae.sample(1, dev)





            # RNN 均值：logβ_det

            g = core.enc(gdev)

            beta_det = core.temporal(seq_inputs, g)               # [T-1]

            logbeta_det = torch.log(beta_det + 1e-8)

            # logβ在2015跳动太剧烈 增加

            # 对 RNN 均值做轻度平滑（一阶/二阶差分，数值可按需调轻/重）

            det_rw1_sigma = 0.35

            det_rw2_sigma = 0.25

            det_rw_weight = 0.7  # 惩罚的权重；想更平滑就再大一点点

            d1_det = logbeta_det[1:] - logbeta_det[:-1]

            pen_det = (d1_det ** 2).mean() / (2 * det_rw1_sigma ** 2)

            if len(logbeta_det) >= 3:

                d2_det = d1_det[1:] - d1_det[:-1]

                pen_det = pen_det + (d2_det ** 2).mean() / (2 * det_rw2_sigma ** 2)



            penalty_acc = penalty_acc + det_rw_weight * pen_det



            # NEW(u-VI) 采样 u，并构造 β

            u_s, _ = q_u.sample(1, dev)                           # [1, T-1]

            u_s = u_s.squeeze(0)                                  # [T-1]

            logbeta = logbeta_det + u_s

            beta_seq = torch.exp(logbeta)                         # [T-1]



            # === ω 动态（det + uω + 全局尺度）===

            # 1) 先得到 det(t)

            omega_det_seq = core.omega_head(seq_inputs, g)  # [T-1] > 0

            logomega_det = torch.log(omega_det_seq + 1e-8)

            # logomega_det = logomega_det - logomega_det.mean()  # 中心化，几何均值=1

            # 平滑正则

            omg_rw1_sigma = 0.35

            omg_rw2_sigma = 0.25

            omg_rw_weight = 0.5

            d1_omg = logomega_det[1:] - logomega_det[:-1]

            pen_omg = (d1_omg ** 2).mean() / (2 * omg_rw1_sigma ** 2)

            if len(logomega_det) >= 3:

                d2_omg = d1_omg[1:] - d1_omg[:-1]

                pen_omg = pen_omg + (d2_omg ** 2).mean() / (2 * omg_rw2_sigma ** 2)

            penalty_acc = penalty_acc + omg_rw_weight * pen_omg



            # 2) uω 残差

            u_omg, _ = q_u_omega.sample(1, dev)

            u_omg = u_omg.squeeze(0)  # [T-1]



            # 3) global 尺度：q_omega 给一个“整体水平”，与 det、u 叠加

            # omega_global = torch.zeros_like(sigmae_s) if not cfg.sirs else q_omega.sample(1, dev)[0]  # 标量>0

            # logomega_global = torch.log(omega_global.squeeze(0) + 1e-8)



            # 4) 组装时间序列

            logomega = logomega_det + u_omg



            omega_seq = torch.exp(logomega) if cfg.sirs else torch.zeros_like(logomega)  # SIR时=0



            gamma_det_seq = core.gamma_head(seq_inputs, g)  # [T-1] > 0

            loggamma_det = torch.log(gamma_det_seq + 1e-8)  # [T-1]

            # loggamma_det = loggamma_det - loggamma_det.mean()  # 中心化，几何均值=1



            # 增加同款惩罚，避免 γ_det 抖动影响可辨识性

            gam_rw1_sigma = 0.35

            gam_rw2_sigma = 0.25

            gam_rw_weight = 0.5  # 可以略低于 beta 的权重

            d1_gam = loggamma_det[1:] - loggamma_det[:-1]

            pen_gam = (d1_gam ** 2).mean() / (2 * gam_rw1_sigma ** 2)

            if len(loggamma_det) >= 3:

                d2_gam = d1_gam[1:] - d1_gam[:-1]

                pen_gam = pen_gam + (d2_gam ** 2).mean() / (2 * gam_rw2_sigma ** 2)

            penalty_acc = penalty_acc + gam_rw_weight * pen_gam

            u_gam, _ = q_u_gamma.sample(1, dev);

            u_gam = u_gam.squeeze(0)  # [T-1]

            loggamma = loggamma_det + u_gam

            gamma_seq = torch.exp(loggamma)  # [T-1]



            # 前向 SIR

            sim_out = core.run_simulation(beta_seq, gamma_seq, omega_seq, node_state0)
            S = sim_out.S
            I = sim_out.I
            R_hat = sim_out.R
            C_hat = sim_out.C

            # 观测：若 sirs=True，用累计 C 对齐你手头“累计移除”的观测；否则仍用 R

            lik_target = C_hat if cfg.sirs else R_hat



            # # 观测似然

            # if cfg.heteroskedastic_obs:

            #     var_t = (sigmae_s ** 2) * (torch.clamp(R_true, min=0.0) + cfg.kappa_obs)

            # else:

            #     var_t = (sigmae_s ** 2) * torch.ones_like(R_true)



            # ------- 观测似然（SIRS: lik_target = C_hat；SIR: = R_hat） -------

            # 选择用于尺度的“基数”：SIRS 用 C 的量级更自然

            base_for_var = torch.clamp(lik_target, min=0.0) if cfg.sirs else torch.clamp(R_true, min=0.0)

            var_t = (sigmae_s ** 2) * (base_for_var + cfg.kappa_obs)



            # 主通道：累计 C（SIRS）或累计 R（SIR）

            log_like_main = (-0.5 * (((lik_target - R_true) ** 2) / var_t + torch.log(var_t) + LOG2PI)).mean()

            log_like = log_like_main



            # 可选辅通道：R_unique（唯一感染累计，用模型的 R_hat 对齐）

            if Ru_true is not None:

                var_t2 = (sigmae_s ** 2) * (torch.clamp(R_hat[:train_T], min=0.0) + cfg.kappa_obs)

                log_like2 = (-0.5 * (

                            ((R_hat[:train_T] - Ru_true) ** 2) / var_t2 + torch.log(var_t2) + LOG2PI)).mean()

                # 等权重合成（需要可调就把 0.5 做成 cfg 参数）

                log_like = 0.5 * (log_like_main + log_like2)



            # log p(θ) + NEW: log p(u)

            log_p_theta = log_lognormal_pdf(sigmae_s, cfg.prior_sigmae_mu, cfg.prior_sigmae_sigma).mean()



            log_p_u = log_prior_u_rw(u_s,

                                     mu0=cfg.prior_u0_mu,

                                     sigma0=cfg.prior_u0_sigma,

                                     tau=cfg.prior_u_rw_sigma)    # 标量



            log_p_u_gamma = log_prior_u_rw(u_gam,

                                           mu0=cfg.prior_u0_mu,

                                           sigma0=cfg.prior_u0_sigma,

                                           tau=max(1e-3, cfg.prior_u_rw_sigma * 0.5))



            # 先验：u_β / u_γ 已有，这里加 u_ω

            log_p_u_omega = log_prior_u_rw(u_omg,

                                           mu0=cfg.prior_u0_mu,

                                           sigma0=cfg.prior_u0_sigma,

                                           tau=max(1e-3, cfg.prior_u_rw_sigma * 0.5))



            # log q(θ) + NEW: log q(u)



            log_q_theta = q_sigmae.log_q(sigmae_s).mean()



            log_q_u = q_u.log_q(u_s)

            log_q_u_gamma = q_u_gamma.log_q(u_gam)

            # 后验：q(u_ω)

            log_q_u_omega = q_u_omega.log_q(u_omg)



            elbo_acc = elbo_acc + (log_like + log_p_theta + log_p_u + log_p_u_gamma + log_p_u_omega

                                   - log_q_theta - log_q_u - log_q_u_gamma - log_q_u_omega)



            # 单调性软约束：sirs=True 时作用在 C（单调），否则作用在 R

            mono_series = lik_target

            penalty_mono = torch.relu(mono_series[:-1] - mono_series[1:]).mean()

            penalty_acc = penalty_acc + cfg.monotonic_weight * penalty_mono



        loss = -(elbo_acc / K) + penalty_acc / K

        loss.backward()

        torch.nn.utils.clip_grad_norm_(params, 5.0)

        opt.step()



        if verbose and ep % 200 == 0:

            print(f"[ep {ep:05d}] loss={loss.item():.4f}  "

                 

                  f"mu_sigmae={q_sigmae.mu.item():.3f}, sigma_sigmae={F.softplus(q_sigmae.rho).item():.3f}  "

                  f"u_beta_sigma(mean)={F.softplus(q_u.rho).mean().item():.3f}  "

                  f"u_gamma_sigma(mean)={F.softplus(q_u_gamma.rho).mean().item():.3f}  "

                  f"u_omega_sigma(mean)={F.softplus(q_u_omega.rho).mean().item():.3f}"

                  )



        if loss.item() < best["loss"]:

            best = {

                "loss": float(loss.item()),

                "state_core": {k: v.detach().cpu() for k, v in core.state_dict().items()},

                "q_snap": dict(

                            sigmae_mu=float(q_sigmae.mu.detach().cpu().item()),

                            sigmae_sigma=float(F.softplus(q_sigmae.rho).detach().cpu().item()),

                            u_mu=q_u.mu.detach().cpu().clone().numpy(),

                            u_sigma=F.softplus(q_u.rho).detach().cpu().clone().numpy(),

                            ugamma_mu=q_u_gamma.mu.detach().cpu().clone().numpy(),

                            ugamma_sigma=F.softplus(q_u_gamma.rho).detach().cpu().clone().numpy(),

                            uomega_mu=q_u_omega.mu.detach().cpu().clone().numpy(),

                            uomega_sigma=F.softplus(q_u_omega.rho).detach().cpu().clone().numpy()

                        )

            }



    # 还原 best

    core.load_state_dict(best["state_core"]); core.eval()

    # NEW: 把 q_u 也回到 best 快照

    q_u.mu.data = torch.tensor(best["q_snap"]["u_mu"], dtype=torch.float32, device=dev)

    q_u.rho.data = torch.log(torch.expm1(torch.tensor(best["q_snap"]["u_sigma"], dtype=torch.float32, device=dev))).clamp_min(-20)

    q_u_gamma.mu.data = torch.tensor(best["q_snap"]["ugamma_mu"], dtype=torch.float32, device=dev)

    q_u_gamma.rho.data = torch.log(torch.expm1(torch.tensor(best["q_snap"]["ugamma_sigma"], dtype=torch.float32, device=dev))).clamp_min(-20)

    q_u_omega.mu.data = torch.tensor(best["q_snap"]["uomega_mu"], dtype=torch.float32, device=dev)

    q_u_omega.rho.data = torch.log(torch.expm1(torch.tensor(best["q_snap"]["uomega_sigma"], dtype=torch.float32, device=dev))).clamp_min(-20)

    # 采样器

    def sample_paths(n_draws: int, T_pred: Optional[int] = None):

        with torch.no_grad():

            T_use = int(train_T if T_pred is None else T_pred)

            R_true_use = torch.tensor(R_obs[:T_use], dtype=torch.float32, device=dev)

            seq_use = build_seq_inputs(dR, nests, T_use, dev)



            S_all, I_all, R_all, B_all = [], [], [], []

            # 额外收集 C

            C_all = []  # 补上这一行

            O_seq_all = []

            G_seq_all = []  # ★ 新增：收集 γ 的时间序列

            for _ in range(n_draws):

                sigmae_s, _, _ = q_sigmae.sample(1, dev)  # 仅用于lik



                g = core.enc(gdev)

                beta_det = core.temporal(seq_use, g)                  # [T-1]

                logbeta_det = torch.log(beta_det + 1e-8)

                # u_s, _ = q_u.sample(1, dev); u_s = u_s.squeeze(0)    # [T-1]

                # logbeta = logbeta_det + u_s

                # beta_seq = torch.exp(logbeta)

                #

                # S, I, R_hat, C_hat = core.sim(beta_seq, S0, I0, R0,

                #                               gamma=gamma_s.squeeze(0),

                #                               omega=omega_s.squeeze(0))

                u_s, _ = q_u.sample(1, dev);

                u_s = u_s.squeeze(0)  # [T-1]

                logbeta = logbeta_det + u_s

                beta_seq = torch.exp(logbeta)



                # === γ 动态（采样时）

                gamma_det_seq = core.gamma_head(seq_use, g)  # [T-1]

                loggamma_det = torch.log(gamma_det_seq + 1e-8)

                # loggamma_det = loggamma_det - loggamma_det.mean()  # 中心化，几何均值=1



                u_gam, _ = q_u_gamma.sample(1, dev);

                u_gam = u_gam.squeeze(0)

                loggamma = loggamma_det + u_gam

                gamma_seq = torch.exp(loggamma)

                G_seq_all.append(gamma_seq.detach().cpu().numpy())  # ★ 新增



                # === ω 动态（采样时）

                omega_det_seq = core.omega_head(seq_use, g)  # [T-1]

                logomega_det = torch.log(omega_det_seq + 1e-8)

                # logomega_det = logomega_det - logomega_det.mean()  # 中心化，几何均值=1



                u_omg, _ = q_u_omega.sample(1, dev);

                u_omg = u_omg.squeeze(0)



                logomega =  logomega_det + u_omg

                omega_seq = torch.exp(logomega) if cfg.sirs else torch.zeros_like(logomega)  # SIR时=0



                sim_out = core.run_simulation(beta_seq, gamma_seq, omega_seq, node_state0)
                S = sim_out.S
                I = sim_out.I
                R_hat = sim_out.R
                C_hat = sim_out.C

                S_all.append(S.detach().cpu().numpy())

                I_all.append(I.detach().cpu().numpy())

                R_all.append(R_hat.detach().cpu().numpy())

                B_all.append(beta_seq.detach().cpu().numpy())

                C_all.append(C_hat.detach().cpu().numpy())



                O_seq_all.append(omega_seq.detach().cpu().numpy())  # 收集 ω 的完整时间序列



        return (np.stack(S_all, 0), np.stack(I_all, 0), np.stack(R_all, 0),

                np.stack(C_all, 0), np.stack(B_all, 0),

                np.stack(O_seq_all, 0), np.stack(G_seq_all, 0))



    return dict(core=core, sample_paths=sample_paths, train_T=train_T)



# ============== 走动验证 ==============

def walk_forward_eval(gdata, years, R_obs, Runiq_obs, dR, nests, cfg: TrainCfg, dev) -> pd.DataFrame:

    T = len(R_obs)

    start = max(cfg.min_train_years, 2)

    end = T

    preds, trues, steps = [], [], []

    for t in range(start, end):

        trainer = train_vi_once(

            gdata, years, R_obs, None, dR, nests, cfg,   # ← Runiq_obs=None

            train_T=t, verbose=False, dev=dev            # ← 传 dev

        )

        S_all, I_all, R_all, C_all, B_all, _,_ = trainer["sample_paths"](n_draws=64, T_pred=t)

        y_pred = (C_all if cfg.sirs else R_all).mean(0)[-1]

        y_true = float(R_obs[t - 1])

        preds.append(y_pred); trues.append(y_true); steps.append(int(t))

    df = pd.DataFrame({"step_train_T": steps, "y_true": trues, "y_pred": preds})

    err = np.array(df["y_pred"]) - np.array(df["y_true"])

    df.attrs["MAE"] = float(np.mean(np.abs(err)))

    df.attrs["RMSE"] = float(np.sqrt(np.mean(err**2)))

    df.attrs["MAPE"] = float(np.mean(np.abs(err) / np.maximum(1.0, np.array(df["y_true"]))))

    return df



# ============== 可视化 & 导出 ==============

# ============== Embedding 诊断导出（新增） ==============

def _extract_node_and_graph_emb(core, gdata, use_pyg: bool):

    """

    从当前 GNN 编码器中提取:

      - node_emb: 第二层激活后的节点级嵌入 [N, hidden]

      - graph_emb: 图级嵌入 [1, hidden] (= node_emb.mean(0, keepdim=True))

    兼容 use_pyg=True/False 两种分支

    """

    core.eval()

    with torch.no_grad():

        if use_pyg and _HAS_PYG and isinstance(gdata, GeometricData):

            x = torch.relu(core.enc.conv1(gdata.x, gdata.edge_index))

            x = torch.relu(core.enc.conv2(x, gdata.edge_index))

            node_emb = x

        else:

            x = torch.relu(core.enc.lin1(gdata['x']))

            x = torch.relu(core.enc.lin2(x))

            node_emb = x

        graph_emb = node_emb.mean(0, keepdim=True)

    return node_emb, graph_emb





def save_embeddings_diagnostics(core, gdata, data_dir: str, out_dir: str, use_pyg: bool = True, max_tsne_n: int = 5000):

    """

    导出节点/图级 embedding 的尽可能详尽的信息:

      1) node_embeddings.csv: 每个节点的 embedding 向量

      2) graph_embedding.csv: 单行，图级 embedding 向量

      3) node_embedding_stats.csv: 每一维的均值/方差/最小/最大

      4) pca_2d.png / pca_explained_variance.csv

      5) tsne_2d.png (节点数很大时随机下采样)

      6) 可选: 与空间坐标/度数的简单相关性 (corr_with_coords.csv)

    """

    import json

    import numpy as np

    import pandas as pd

    import matplotlib.pyplot as plt

    from sklearn.decomposition import PCA

    from sklearn.manifold import TSNE



    os.makedirs(out_dir, exist_ok=True)



    # 设备放到和模型一致

    dev = next(core.parameters()).device

    # 统一把 gdata 放到同一设备

    if use_pyg and _HAS_PYG and isinstance(gdata, GeometricData):

        gdev = gdata.to(dev)

        N = gdev.x.size(0)

    else:

        gdev = {k: v.to(dev) for k, v in gdata.items()}

        N = gdev['x'].size(0)



    # 1) 取出节点/图级 embedding

    node_emb_t, graph_emb_t = _extract_node_and_graph_emb(core, gdev, use_pyg=use_pyg)

    node_emb = node_emb_t.detach().cpu().numpy()   # [N, H]

    graph_emb = graph_emb_t.detach().cpu().numpy() # [1, H]

    H = node_emb.shape[1]



    # 2) 保存 CSV

    df_nodes = pd.DataFrame(node_emb, columns=[f"dim_{i}" for i in range(H)])

    df_nodes.insert(0, "node_idx", np.arange(N, dtype=int))

    df_nodes.to_csv(os.path.join(out_dir, "node_embeddings.csv"), index=False, float_format="%.6g")



    df_graph = pd.DataFrame([graph_emb.squeeze()], columns=[f"dim_{i}" for i in range(H)])

    df_graph.to_csv(os.path.join(out_dir, "graph_embedding.csv"), index=False, float_format="%.6g")



    # 3) 统计信息（每一维）

    stats = []

    for i in range(H):

        col = node_emb[:, i]

        stats.append(dict(

            dim=i, mean=float(col.mean()), std=float(col.std(ddof=1)),

            min=float(col.min()), max=float(col.max()),

            q2p5=float(np.percentile(col, 2.5)),

            q25=float(np.percentile(col, 25)),

            median=float(np.percentile(col, 50)),

            q75=float(np.percentile(col, 75)),

            q97p5=float(np.percentile(col, 97.5))

        ))

    pd.DataFrame(stats).to_csv(os.path.join(out_dir, "node_embedding_stats.csv"), index=False, float_format="%.6g")



    # 4) PCA 降维与图片

    pca = PCA(n_components=2, random_state=42)

    node_2d = pca.fit_transform(node_emb)

    pd.DataFrame(dict(PC1=node_2d[:,0], PC2=node_2d[:,1], node_idx=np.arange(N))).to_csv(

        os.path.join(out_dir, "pca_2d_points.csv"), index=False, float_format="%.6g"

    )

    pd.DataFrame(dict(component=[1,2], explained_variance_ratio=pca.explained_variance_ratio_)).to_csv(

        os.path.join(out_dir, "pca_explained_variance.csv"), index=False, float_format="%.6g"

    )

    plt.figure(figsize=(6.2,6))

    plt.scatter(node_2d[:,0], node_2d[:,1], s=6, alpha=0.7)

    plt.title("Node embeddings (PCA 2D)")

    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "pca_2d.png"), dpi=180)

    plt.close()



    # 5) t-SNE（节点太多时下采样，避免极慢）

    idx = np.arange(N)

    if N > max_tsne_n:

        rng = np.random.default_rng(42)

        idx = rng.choice(N, size=max_tsne_n, replace=False)

    node_tsne = TSNE(n_components=2, random_state=42, init="pca", perplexity=30).fit_transform(node_emb[idx])

    pd.DataFrame(dict(tSNE1=node_tsne[:,0], tSNE2=node_tsne[:,1], node_idx=idx)).to_csv(

        os.path.join(out_dir, "tsne_2d_points.csv"), index=False, float_format="%.6g"

    )

    plt.figure(figsize=(6.2,6))

    plt.scatter(node_tsne[:,0], node_tsne[:,1], s=6, alpha=0.7)

    plt.title(f"Node embeddings (t-SNE 2D)  n={len(idx)}")

    plt.xlabel("tSNE1"); plt.ylabel("tSNE2"); plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "tsne_2d.png"), dpi=180)

    plt.close()



    # 6) 与空间坐标/度数的简单相关性（可帮助解释 embedding 是否在编码密度/拓扑）

    # 6.1 读取 nodes.csv 的坐标

    corr_rows = []

    try:

        nodes_csv = os.path.join(data_dir, "nodes.csv")

        nodes_df = pd.read_csv(nodes_csv)

        # 坐标列智能匹配

        cols_lower = [c.lower() for c in nodes_df.columns]

        def pick(cands):

            for c in cands:

                if c in cols_lower:

                    return nodes_df.columns[cols_lower.index(c)]

            return None

        x_col = pick(['easting','eastings','x','lon','longitude'])

        y_col = pick(['northing','northings','y','lat','latitude'])

        xs = nodes_df[x_col].to_numpy() if x_col else None

        ys = nodes_df[y_col].to_numpy() if y_col else None

    except Exception as e:

        xs, ys = None, None



    # 6.2 度数（degree）

    try:

        import numpy as np

        if use_pyg and _HAS_PYG and isinstance(gdev, GeometricData):

            ei = gdev.edge_index.detach().cpu().numpy()

        else:

            ei = gdev['edge_index'].detach().cpu().numpy()

        deg = np.bincount(ei.reshape(-1), minlength=N)

    except Exception:

        deg = None



    # 6.3 计算相关系数（与每一维 embedding）

    import math

    def safe_corr(a, b):

        if a is None or b is None: return np.nan

        if len(a) != len(b): return np.nan

        if np.std(a) < 1e-12 or np.std(b) < 1e-12: return np.nan

        return float(np.corrcoef(a, b)[0,1])



    for i in range(H):

        col = node_emb[:, i]

        corr_rows.append(dict(

            dim=i,

            corr_with_x=safe_corr(xs, col),

            corr_with_y=safe_corr(ys, col),

            corr_with_degree=safe_corr(deg, col)

        ))

    pd.DataFrame(corr_rows).to_csv(os.path.join(out_dir, "corr_with_coords.csv"),

                                   index=False, float_format="%.6g")



    # 7) 再补充一个“图级 embedding 的整体量级”小报告

    report = dict(

        N_nodes=int(N),

        H_dim=int(H),

        graph_emb_L2=float(np.linalg.norm(graph_emb)),

        graph_emb_mean=float(graph_emb.mean()),

        graph_emb_std=float(graph_emb.std()),

        node_emb_L2_mean=float(np.linalg.norm(node_emb, axis=1).mean()),

        node_emb_L2_std=float(np.linalg.norm(node_emb, axis=1).std(ddof=1))

    )

    with open(os.path.join(out_dir, "embedding_report.json"), "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)



    print("[Embedding] 已导出：node_embeddings.csv, graph_embedding.csv, node_embedding_stats.csv,",

          "pca_2d.png, pca_2d_points.csv, pca_explained_variance.csv, tsne_2d.png, tsne_2d_points.csv,",

          "corr_with_coords.csv, embedding_report.json 到", out_dir)





def plot_band(ax, x, mean, q25, q75, q2p5, q97p5, label, color="#1f77b4"):

    ax.plot(x, q97p5, linestyle=(0,(3,3,1,3)), color=color, linewidth=1.2)

    ax.plot(x, q2p5,  linestyle=(0,(3,3,1,3)), color=color, linewidth=1.2)

    ax.plot(x, q75, linestyle="--", color=color, linewidth=1.5)

    ax.plot(x, q25, linestyle="--", color=color, linewidth=1.5)

    std_est = (q75 - q25) / 1.349

    ax.fill_between(x, mean - std_est, mean + std_est, color=color, alpha=0.15, linewidth=0)

    ax.plot(x, mean, color=color, linewidth=2.2, label=label)



def run_all(

    data_dir: str,

    out_dir: str = "vi_out",

    epochs: int = 2500,

    hidden: int = 64,

    lr: float = 3e-3,

    weight_decay: float = 1e-5,

    no_pyg: bool = False,

    sirs: bool = False,

    vi_mc_samples: int = 4,

    vi_draws_for_paths: int = 200,

    heteroskedastic_obs: int = 1,

    kappa_obs: float = 50.0,

    # 先验

    prior_gamma_mu: float = float(np.log(1.0)),

    prior_gamma_sigma: float = 0.7,

    prior_sigmae_mu: float = float(np.log(0.5)),

    prior_sigmae_sigma: float = 0.7,

    prior_omega_mu: float = float(np.log(0.5)),

    prior_omega_sigma: float = 0.7,

    # NEW(u-VI) 先验

    prior_u0_mu: float = 0.0,

    prior_u0_sigma: float = 0.5,

    prior_u_rw_sigma: float = 0.30,

    # 单调

    monotonic_weight: float = 1.0,

    # N 与 I0

    N_base: float = 40000.0,

    allow_N_perturb: int = 0,

    N_low_ratio: float = 0.9,

    N_high_ratio: float = 1.1,

    I0_scale_low: float = 0.8,

    I0_scale_high: float = 1.5,

    # 走动验证

    do_walk_forward: int = 1,

    min_train_years: int = 3

):

    os.makedirs(out_dir, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # gdata, years, R_obs, dR, nests, _ = load_graph_and_series(data_dir, no_pyg=no_pyg)

    gdata, years, R_obs, Runiq_obs, dR, nests, _ = load_graph_and_series(data_dir, no_pyg=no_pyg)



    cfg = TrainCfg(

        epochs=epochs, lr=lr, weight_decay=weight_decay, hidden=hidden,

        monthly_steps=12, use_pyg=(not no_pyg), sirs=sirs,

        N_base=N_base, allow_N_perturb=bool(allow_N_perturb),

        N_low_ratio=N_low_ratio, N_high_ratio=N_high_ratio,

        I0_scale_low=I0_scale_low, I0_scale_high=I0_scale_high,

        heteroskedastic_obs=bool(heteroskedastic_obs), kappa_obs=kappa_obs,

        prior_gamma_mu=prior_gamma_mu, prior_gamma_sigma=prior_gamma_sigma,

        prior_sigmae_mu=prior_sigmae_mu, prior_sigmae_sigma=prior_sigmae_sigma,

        prior_omega_mu=prior_omega_mu, prior_omega_sigma=prior_omega_sigma,

        vi_mc_samples=vi_mc_samples, vi_draws_for_paths=vi_draws_for_paths,

        prior_u0_mu=prior_u0_mu, prior_u0_sigma=prior_u0_sigma, prior_u_rw_sigma=prior_u_rw_sigma,

        monotonic_weight=monotonic_weight,

        do_walk_forward=bool(do_walk_forward), min_train_years=min_train_years,

        out_dir=out_dir

    )



    # 训练（全量）

    # trainer = train_vi_once(gdata, years, R_obs, dR, nests, cfg, train_T=None, verbose=True)

    # 全量训练

    trainer = train_vi_once(

        gdata, years, R_obs, Runiq_obs, dR, nests, cfg,

        train_T=None, verbose=True, dev=dev

    )

    S_all, I_all, R_all, C_all, B_all, O_seq_all,G_seq_all  = trainer["sample_paths"](n_draws=cfg.vi_draws_for_paths,

                                                                                T_pred=None)

    # —— 新增：导出 GNN 节点/图级 embedding 的详细诊断 —— #

    save_embeddings_diagnostics(trainer["core"], gdata, data_dir=data_dir, out_dir=os.path.join(out_dir, "embeddings"),

                                use_pyg=(not no_pyg))



    S_stat = quantile_stats(S_all, axis=0)

    I_stat = quantile_stats(I_all, axis=0)

    R_stat = quantile_stats(R_all, axis=0)

    C_stat = quantile_stats(C_all, axis=0)

    B_stat = quantile_stats(B_all, axis=0)

    logB_stat = quantile_stats(np.log(B_all + 1e-12) - np.log(cfg.N_base), axis=0)

    O_stat = quantile_stats(O_seq_all, axis=0)   # [draw, T-1] → 统计到每年

    # ★ 逐年 γ 统计（新增）

    G_stat = quantile_stats(G_seq_all, axis=0)

    # 导出 CSV

    years_arr = np.asarray(years)

    df_O = pd.DataFrame({

        "year": years_arr[1:],

        "omega_mean": O_stat["mean"], "omega_q25": O_stat["q25"], "omega_q75": O_stat["q75"],

        "omega_q2p5": O_stat["q2p5"], "omega_q97p5": O_stat["q97p5"],

        "immune_duration_mean": 1.0 / np.maximum(O_stat["mean"], 1e-12),

    })

    df_O.to_csv(os.path.join(out_dir, "vi_summary_omega.csv"), index=False)



    # ★ 新增：逐年 γ 的后验带

    df_G = pd.DataFrame({

        "year": years_arr[1:],  # γ_t 与 β/ω 一样是长度 T-1

        "gamma_mean": G_stat["mean"],

        "gamma_q25": G_stat["q25"], "gamma_q75": G_stat["q75"],

        "gamma_q2p5": G_stat["q2p5"], "gamma_q97p5": G_stat["q97p5"],

    })

    df_G.to_csv(os.path.join(out_dir, "vi_summary_gamma.csv"), index=False)



    symbolic_models = []

    try:

        transition_features = build_transition_feature_matrix(

            years=years_arr,

            S=S_stat["mean"],

            I=I_stat["mean"],

            R=R_stat["mean"],

            C=C_stat["mean"],

            dR=dR[:len(years_arr)],

            nests=nests[:len(years_arr)],

            N=cfg.N_base,

            include_trigonometric=False,

            include_exponential=True,

        )



        beta_model = fit_symbolic_model(

            target_name="β",

            y=B_stat["mean"],

            features=transition_features,

        )

        symbolic_models.append(beta_model)



        gamma_model = fit_symbolic_model(

            target_name="γ",

            y=G_stat["mean"],

            features=transition_features,

        )

        symbolic_models.append(gamma_model)



        if cfg.sirs:

            omega_values = np.clip(O_stat["mean"], 1e-10, None)

            omega_model = fit_symbolic_model(

                target_name="ω",

                y=omega_values,

                features=transition_features,

            )

        else:

            omega_model = SymbolicModel(

                target_name="ω",

                intercept=float(np.log(1e-12)),

                coefficients={},

                feature_order=transition_features.names,

                support=[],

                r2_log=0.0,

                residual_std_log=0.0,

                y_mean=0.0,

            )

        symbolic_models.append(omega_model)



        export_symbolic_summary(

            symbolic_models,

            transition_features,

            out_dir=os.path.join(out_dir, "symbolic"),

        )

        print(f"[Symbolic] 输出 β/γ/ω 公式至 {os.path.join(out_dir, 'symbolic')}")

    except Exception as exc:

        print(f"[Symbolic] 生成公式失败：{exc}")



    df_R = pd.DataFrame({

        "year": years_arr,

        "R_mean": R_stat["mean"], "R_q25": R_stat["q25"], "R_q75": R_stat["q75"],

        "R_q2p5": R_stat["q2p5"], "R_q97p5": R_stat["q97p5"],

        "R_obs": Runiq_obs  # 注意：观测仍是累计移除；在 SIRS 下你真正拟合的是 C，但这里保留对照

    })



    df_C = pd.DataFrame({

        "year": years_arr,

        "C_mean": C_stat["mean"], "C_q25": C_stat["q25"], "C_q75": C_stat["q75"],

        "C_q2p5": C_stat["q2p5"], "C_q97p5": C_stat["q97p5"],

        "C_obs": R_obs  # 观测“累计移除”→ 作为 C_obs

    })



    df_S = pd.DataFrame({

        "year": years_arr,

        "S_mean": S_stat["mean"], "S_q25": S_stat["q25"], "S_q75": S_stat["q75"],

        "S_q2p5": S_stat["q2p5"], "S_q97p5": S_stat["q97p5"]

    })

    df_I = pd.DataFrame({

        "year": years_arr,

        "I_mean": I_stat["mean"], "I_q25": I_stat["q25"], "I_q75": I_stat["q75"],

        "I_q2p5": I_stat["q2p5"], "I_q97p5": I_stat["q97p5"]

    })

    df_B = pd.DataFrame({

        "year": years_arr[1:],

        "beta_mean": B_stat["mean"], "beta_q25": B_stat["q25"], "beta_q75": B_stat["q75"],

        "beta_q2p5": B_stat["q2p5"], "beta_q97p5": B_stat["q97p5"],

        "logbeta_mean": logB_stat["mean"], "logbeta_q25": logB_stat["q25"],

        "logbeta_q75": logB_stat["q75"], "logbeta_q2p5": logB_stat["q2p5"],

        "logbeta_q97p5": logB_stat["q97p5"]

    })





    # df_R.to_csv(os.path.join(out_dir, "vi_summary_R.csv"), index=False)

    df_C.to_csv(os.path.join(out_dir, "vi_summary_C.csv"), index=False)

    df_R.to_csv(os.path.join(out_dir, "vi_summary_R.csv"), index=False)

    df_S.to_csv(os.path.join(out_dir, "vi_summary_S.csv"), index=False)

    df_I.to_csv(os.path.join(out_dir, "vi_summary_I.csv"), index=False)

    df_B.to_csv(os.path.join(out_dir, "vi_summary_beta.csv"), index=False)



    # 作图

    def save_figs():

        # 1) 累计通道：SIRS→C 对齐观测；SIR→R 对齐观测

        if sirs:

            fig, ax = plt.subplots(figsize=(7.2, 4.2))

            plot_band(ax, years_arr, C_stat["mean"], C_stat["q25"], C_stat["q75"], C_stat["q2p5"], C_stat["q97p5"],

                      "C(t)")

            ax.scatter(years_arr, R_obs, s=28, c="orange", label="Observed (cum. removed ≈ C)", zorder=5)

            ax.set_title("C(t) (VI posterior)");

            ax.set_xlabel("Year");

            ax.set_ylabel("Trees")

            ax.grid(alpha=0.25);

            ax.legend();

            fig.tight_layout()

            fig.savefig(os.path.join(out_dir, "vi_fig_C.png"), dpi=200);

            plt.close(fig)

        else:

            fig, ax = plt.subplots(figsize=(7.2, 4.2))

            plot_band(ax, years_arr, R_stat["mean"], R_stat["q25"], R_stat["q75"], R_stat["q2p5"], R_stat["q97p5"],

                      "R(t)")

            ax.scatter(years_arr, R_obs, s=28, c="orange", label="Observed", zorder=5)

            ax.set_title("R(t) (VI posterior)");

            ax.set_xlabel("Year");

            ax.set_ylabel("Trees")

            ax.grid(alpha=0.25);

            ax.legend();

            fig.tight_layout()

            fig.savefig(os.path.join(out_dir, "vi_fig_R.png"), dpi=200);

            plt.close(fig)



        # 2) 若存在 R_unique：额外对比 R(t) vs R_unique（统一用 R_stat）

        try:

            ys_tmp = pd.read_csv(os.path.join(data_dir, "yearly_series.csv"))

            if "R_unique" in ys_tmp.columns:

                fig, ax = plt.subplots(figsize=(7.2, 4.2))

                plot_band(ax, years_arr, R_stat["mean"], R_stat["q25"], R_stat["q75"], R_stat["q2p5"], R_stat["q97p5"],

                          "R(t)")

                ax.scatter(years_arr, ys_tmp["R_unique"].values, s=28, c="tab:orange", label="Observed R_unique",

                           zorder=5)

                ax.set_title("R(t) vs Observed R_unique");

                ax.set_xlabel("Year");

                ax.set_ylabel("Trees")

                ax.grid(alpha=0.25);

                ax.legend();

                fig.tight_layout()

                fig.savefig(os.path.join(out_dir, "vi_fig_R_unique.png"), dpi=200);

                plt.close(fig)

        except Exception:

            pass



        # 3) 无论 SIRS 与否，都画出 S(t)、I(t)、logβ(t)

        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        plot_band(ax, years_arr, S_stat["mean"], S_stat["q25"], S_stat["q75"], S_stat["q2p5"], S_stat["q97p5"], "S(t)")

        ax.set_title("S(t)");

        ax.set_xlabel("Year");

        ax.set_ylabel("Trees");

        ax.grid(alpha=0.25)

        fig.tight_layout();

        fig.savefig(os.path.join(out_dir, "vi_fig_S.png"), dpi=200);

        plt.close(fig)



        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        plot_band(ax, years_arr, I_stat["mean"], I_stat["q25"], I_stat["q75"], I_stat["q2p5"], I_stat["q97p5"], "I(t)")

        ax.set_title("I(t)");

        ax.set_xlabel("Year");

        ax.set_ylabel("Trees");

        ax.grid(alpha=0.25)

        fig.tight_layout();

        fig.savefig(os.path.join(out_dir, "vi_fig_I.png"), dpi=200);

        plt.close(fig)



        fig, ax = plt.subplots(figsize=(7.2, 4.2))

        plot_band(ax, years_arr[1:], logB_stat["mean"], logB_stat["q25"], logB_stat["q75"], logB_stat["q2p5"],

                  logB_stat["q97p5"], "log β(t)")

        ax.set_title("log β(t)");

        ax.set_xlabel("Year");

        ax.set_ylabel("log β");

        ax.grid(alpha=0.25)

        fig.tight_layout();

        fig.savefig(os.path.join(out_dir, "vi_fig_logbeta.png"), dpi=200);

        plt.close(fig)

    save_figs()



    # 走动验证

    if do_walk_forward:

        # df_cv = walk_forward_eval(gdata, years, R_obs, dR, nests, cfg)

        df_cv = walk_forward_eval(gdata, years, R_obs, Runiq_obs, dR, nests, cfg, dev)

        df_cv.to_csv(os.path.join(out_dir, "walk_forward_cv.csv"), index=False)

        with open(os.path.join(out_dir, "walk_forward_scores.txt"), "w", encoding="utf-8") as f:

            f.write(f"MAE={df_cv.attrs['MAE']:.4f}\nRMSE={df_cv.attrs['RMSE']:.4f}\nMAPE={df_cv.attrs['MAPE']:.4f}\n")



    print(f"\n[VI β=mean+u] 输出目录：{os.path.abspath(out_dir)}")

    return dict(out_dir=out_dir, years=years, R_obs=R_obs)



# ============== CLI ==============

def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="cleaned_richmond")

    ap.add_argument("--out_dir", type=str, default="vi_out_sirs")



    ap.add_argument("--epochs", type=int, default=2500)

    ap.add_argument("--hidden", type=int, default=64)

    ap.add_argument("--lr", type=float, default=3e-3)

    ap.add_argument("--weight_decay", type=float, default=1e-5)

    ap.add_argument("--no_pyg", action="store_true")

    ap.add_argument("--sirs", action="store_true")



    # VI

    ap.add_argument("--vi_mc_samples", type=int, default=4)

    ap.add_argument("--vi_draws_for_paths", type=int, default=200)



    # 观测

    ap.add_argument("--heteroskedastic_obs", type=int, default=1)

    ap.add_argument("--kappa_obs", type=float, default=50.0)



    # 先验（γ/σ_e/ω）

    ap.add_argument("--prior_gamma_mu", type=float, default=float(np.log(1.0)))

    ap.add_argument("--prior_gamma_sigma", type=float, default=0.7)

    ap.add_argument("--prior_sigmae_mu", type=float, default=float(np.log(0.5)))

    ap.add_argument("--prior_sigmae_sigma", type=float, default=0.7)

    ap.add_argument("--prior_omega_mu", type=float, default=float(np.log(0.5)))

    ap.add_argument("--prior_omega_sigma", type=float, default=0.7)



    # NEW(u-VI) 先验

    ap.add_argument("--prior_u0_mu", type=float, default=0.0)

    ap.add_argument("--prior_u0_sigma", type=float, default=0.5)

    ap.add_argument("--prior_u_rw_sigma", type=float, default=0.30)



    # 单调 & N/I0

    ap.add_argument("--monotonic_weight", type=float, default=1.0)

    ap.add_argument("--N_base", type=float, default=40000.0)

    ap.add_argument("--allow_N_perturb", type=int, default=0)

    ap.add_argument("--N_low_ratio", type=float, default=0.9)

    ap.add_argument("--N_high_ratio", type=float, default=1.1)

    ap.add_argument("--I0_scale_low", type=float, default=0.8)

    ap.add_argument("--I0_scale_high", type=float, default=1.5)



    # 走动验证

    ap.add_argument("--do_walk_forward", type=int, default=1)

    ap.add_argument("--min_train_years", type=int, default=3)



    args = ap.parse_args()



    run_all(

        data_dir=args.data_dir,

        out_dir=args.out_dir,

        epochs=args.epochs,

        hidden=args.hidden,

        lr=args.lr,

        weight_decay=args.weight_decay,

        no_pyg=args.no_pyg,

        sirs=args.sirs,

        vi_mc_samples=args.vi_mc_samples,

        vi_draws_for_paths=args.vi_draws_for_paths,

        heteroskedastic_obs=args.heteroskedastic_obs,

        kappa_obs=args.kappa_obs,

        prior_gamma_mu=args.prior_gamma_mu,

        prior_gamma_sigma=args.prior_gamma_sigma,

        prior_sigmae_mu=args.prior_sigmae_mu,

        prior_sigmae_sigma=args.prior_sigmae_sigma,

        prior_omega_mu=args.prior_omega_mu,

        prior_omega_sigma=args.prior_omega_sigma,

        prior_u0_mu=args.prior_u0_mu,

        prior_u0_sigma=args.prior_u0_sigma,

        prior_u_rw_sigma=args.prior_u_rw_sigma,

        monotonic_weight=args.monotonic_weight,

        N_base=args.N_base,

        allow_N_perturb=args.allow_N_perturb,

        N_low_ratio=args.N_low_ratio,

        N_high_ratio=args.N_high_ratio,

        I0_scale_low=args.I0_scale_low,

        I0_scale_high=args.I0_scale_high,

        do_walk_forward=args.do_walk_forward,

        min_train_years=args.min_train_years

    )



if __name__ == "__main__":

    main()

