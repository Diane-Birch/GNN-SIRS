"""
Neural message-passing SIRS model over individual trees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_feats(coords: torch.Tensor) -> torch.Tensor:
    mean = coords.mean(dim=0, keepdim=True)
    std = coords.std(dim=0, keepdim=True).clamp_min(1.0)
    return (coords - mean) / std


def _aggregate_messages(edge_index: torch.LongTensor, src_values: torch.Tensor, num_nodes: int) -> torch.Tensor:
    src, dst = edge_index
    agg = torch.zeros(num_nodes, device=src_values.device, dtype=src_values.dtype)
    agg.index_add_(0, dst, src_values[src])
    return agg


@dataclass
class NodeSIRSSummary:
    S: torch.Tensor
    I: torch.Tensor
    R: torch.Tensor
    infection_prob: torch.Tensor
    event_prob: torch.Tensor
    new_infections: torch.Tensor


class NodeSIRSModel(nn.Module):
    """
    Differentiable SIRS simulator with graph-based infection pressure.
    """

    def __init__(self, node_feat_dim: int, hidden: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(node_feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.beta_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.event_head = nn.Sequential(
            nn.Linear(hidden + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.gamma_param = nn.Parameter(torch.tensor(0.0))
        self.omega_param = nn.Parameter(torch.tensor(-3.0))

    @property
    def gamma(self) -> torch.Tensor:
        return torch.sigmoid(self.gamma_param)

    @property
    def omega(self) -> torch.Tensor:
        return torch.sigmoid(self.omega_param)

    def forward(
        self,
        coords: torch.Tensor,
        edge_index: torch.LongTensor,
        steps: int,
        event_targets: Optional[torch.Tensor] = None,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[Optional[torch.Tensor], NodeSIRSSummary]:
        """
        Simulate SIRS dynamics and optionally compute BCE loss against observed events.

        Parameters
        ----------
        coords:
            Node coordinate features (N, 2).
        edge_index:
            Edge list (2, E) with src/dst indices.
        steps:
            Number of temporal steps to unroll.
        event_targets:
            Optional tensor (steps, N) of binary indicators (0/1) for observed infections.
        initial_state:
            Optional tuple (S0, I0, R0); if omitted, defaults to (1, 0, 0).
        """

        device = coords.device
        N = coords.size(0)

        feats = _normalize_feats(coords)
        node_emb = self.encoder(feats)

        if initial_state is None:
            S_t = torch.ones(N, device=device)
            I_t = torch.zeros(N, device=device)
            R_t = torch.zeros(N, device=device)
        else:
            S_t, I_t, R_t = initial_state

        S_all = []
        I_all = []
        R_all = []
        new_inf_all = []
        infection_prob_all = []
        event_prob_all = []

        loss = None
        if event_targets is not None:
            loss = torch.zeros((), device=device)

        for t in range(steps):
            agg_I = _aggregate_messages(edge_index, I_t, N).unsqueeze(-1)
            beta_input = torch.cat([node_emb, agg_I], dim=-1)
            beta = F.softplus(self.beta_head(beta_input)).squeeze(-1) + 1e-4
            infection_prob = 1.0 - torch.exp(-beta)

            event_logit = self.event_head(beta_input).squeeze(-1)
            event_prob = torch.sigmoid(event_logit)

            new_infections = S_t * infection_prob
            recover = self.gamma * I_t
            return_s = self.omega * R_t

            S_next = torch.clamp(S_t - new_infections + return_s, 0.0, 1.0)
            I_next = torch.clamp(I_t + new_infections - recover, 0.0, 1.0)
            R_next = torch.clamp(R_t + recover - return_s, 0.0, 1.0)

            S_all.append(S_next)
            I_all.append(I_next)
            R_all.append(R_next)
            new_inf_all.append(new_infections)
            infection_prob_all.append(infection_prob)
            event_prob_all.append(event_prob)

            if event_targets is not None:
                target_t = event_targets[t]
                loss = loss + F.binary_cross_entropy(event_prob, target_t, reduction="mean")

            S_t, I_t, R_t = S_next, I_next, R_next

        summary = NodeSIRSSummary(
            S=torch.stack(S_all, dim=0),
            I=torch.stack(I_all, dim=0),
            R=torch.stack(R_all, dim=0),
            infection_prob=torch.stack(infection_prob_all, dim=0),
            event_prob=torch.stack(event_prob_all, dim=0),
            new_infections=torch.stack(new_inf_all, dim=0),
        )
        return loss, summary


def simulate_paths(
    model: NodeSIRSModel,
    coords: torch.Tensor,
    edge_index: torch.LongTensor,
    steps: int,
) -> NodeSIRSSummary:
    model.eval()
    with torch.no_grad():
        _, summary = model(coords, edge_index, steps, event_targets=None)
    return summary

