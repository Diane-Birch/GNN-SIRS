"""
LaGNA-inspired utilities for interpretable SIRS parameter inference.

This package hosts helper modules that modularise the Richmond training script:

- feature_library: construct candidate basis functions from simulated S/I/R states
  and auxiliary covariates (inspired by Gao et al. 2024 LaGNA framework).
- symbolic_regression: fit sparse, human-readable formulas for β/γ/ω.
- reporting: utility to turn fitted models into text/JSON exports.
"""

from .feature_library import build_transition_feature_matrix, TransitionFeatures
from .symbolic_regression import fit_symbolic_model, SymbolicModel
from .reporting import export_symbolic_summary

__all__ = [
    "build_transition_feature_matrix",
    "TransitionFeatures",
    "fit_symbolic_model",
    "SymbolicModel",
    "export_symbolic_summary",
]

