"""
Node-level SIRS diffusion modelling utilities.

This package provides:
  - data loading helpers that transform the RichmondFullData.csv file into
    per-node temporal tensors and adjacency matrices;
  - a differentiable node-level SIRS model with message passing infection;
  - training / inference helpers that recover infection probabilities and
    estimated diffusion paths.
"""

from .data import load_richmond_node_series, build_radius_graph
from .model import NodeSIRSModel, NodeSIRSSummary, simulate_paths
from .training import train_node_sirs

__all__ = [
    "load_richmond_node_series",
    "build_radius_graph",
    "NodeSIRSModel",
    "NodeSIRSSummary",
    "simulate_paths",
    "train_node_sirs",
]

