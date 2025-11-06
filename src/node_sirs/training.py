"""
Training utilities for the node-level SIRS model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .data import RichmondNodeSeries, build_radius_graph
from .model import NodeSIRSModel, NodeSIRSSummary


@dataclass
class TrainingOutput:
    model: NodeSIRSModel
    summary: NodeSIRSSummary
    edge_index: torch.LongTensor
    coords: torch.Tensor
    years: np.ndarray
    loss_history: torch.Tensor
    events: torch.Tensor


def train_node_sirs(
    dataset: RichmondNodeSeries,
    radius: float,
    *,
    hidden: int = 64,
    epochs: int = 500,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: Optional[str] = None,
) -> TrainingOutput:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    edge_index_np, _ = build_radius_graph(dataset.coords, radius=radius)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    coords = torch.tensor(dataset.coords, dtype=torch.float32, device=device)

    # events: (T, N)
    events = torch.tensor(dataset.events.T, dtype=torch.float32, device=device)
    steps, N = events.shape

    model = NodeSIRSModel(node_feat_dim=coords.size(1), hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_history = []
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss, summary = model(coords, edge_index, steps, event_targets=events)
        if loss is None:
            raise RuntimeError("Loss should not be None during training.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        loss_history.append(float(loss.detach().cpu()))
        if epoch % 50 == 0 or epoch == 1:
            print(f"[epoch {epoch:04d}] loss={loss.item():.4f} "
                  f"gamma={model.gamma.item():.3f} omega={model.omega.item():.3f}")

    model.eval()
    with torch.no_grad():
        _, summary = model(coords, edge_index, steps, event_targets=None)

    return TrainingOutput(
        model=model,
        summary=summary,
        edge_index=edge_index,
        coords=coords,
        years=dataset.years,
        loss_history=torch.tensor(loss_history),
        events=events,
    )

