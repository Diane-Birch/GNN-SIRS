"""
Feature construction utilities inspired by the LaGNA framework.

The idea is to expose an explicit library of candidate basis functions built from
the (S, I, R, C) state trajectories and auxiliary covariates.  Instead of relying
on a black-box GRU, we later fit sparse linear models (in log-space) to derive
human-readable formulas for β(t), γ(t) and ω(t).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np


@dataclass
class TransitionFeatures:
    """
    Container for transition-level feature matrix.

    Attributes
    ----------
    matrix:
        Array with shape [T-1, F] containing the engineered features for each
        year-to-year transition.
    names:
        List of feature names aligned with the columns of ``matrix``.
    years:
        Year indices associated with each transition (aligned with ``matrix`` rows).
    S_mid/I_mid/R_mid/C_mid:
        Mid-point states used to build the features (``0.5 * (x_t + x_{t+1})``).
    dR:
        Observed yearly increments aligned with the transitions.
    nests:
        Reported nest removals aligned with the transitions.
    metadata:
        Optional additional diagnostic information (e.g. scaling constants).
    """

    matrix: np.ndarray
    names: List[str]
    years: np.ndarray
    S_mid: np.ndarray
    I_mid: np.ndarray
    R_mid: np.ndarray
    C_mid: np.ndarray
    dR: np.ndarray
    nests: np.ndarray
    metadata: Dict[str, np.ndarray]


def _safe_ratio(x: np.ndarray, denom: float, eps: float = 1e-8) -> np.ndarray:
    return x / max(denom, eps)


def _normalise(x: np.ndarray, denom: float) -> np.ndarray:
    return x / (denom + 1e-8)


def _add_feature(features: List[np.ndarray], names: List[str], name: str, values: np.ndarray):
    if values.ndim != 1:
        raise ValueError(f"feature '{name}' must be 1D, got shape {values.shape}")
    features.append(values.astype(np.float64))
    names.append(name)


def build_transition_feature_matrix(
    *,
    years: np.ndarray,
    S: np.ndarray,
    I: np.ndarray,
    R: np.ndarray,
    C: np.ndarray,
    dR: np.ndarray,
    nests: np.ndarray,
    N: float,
    include_trigonometric: bool = False,
    include_exponential: bool = True,
) -> TransitionFeatures:
    """
    Construct a feature matrix for year-to-year transitions.

    Parameters
    ----------
    years:
        Array of observation years (length ``T``).  Transition features are aligned
        with ``years[1:]``.
    S, I, R, C:
        Posterior means (or any representative trajectories) of the SIRS states
        with shape ``[T]``.
    dR, nests:
        Observed yearly increments and nest counts with shape ``[T]``.
    N:
        Population size used to normalise the states; prevents feature magnitude
        blow-up and mirrors the LaGNA practice of working with dimensionless inputs.
    include_trigonometric / include_exponential:
        Optional switches to add light-weight non-polynomial features, mimicking
        LaGNA's richer basis while keeping the library compact (important because
        the Richmond series is short).
    """

    years = np.asarray(years, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    I = np.asarray(I, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    C = np.asarray(C, dtype=np.float64)
    dR = np.asarray(dR, dtype=np.float64)
    nests = np.asarray(nests, dtype=np.float64)

    if any(arr.ndim != 1 for arr in (years, S, I, R, C, dR, nests)):
        raise ValueError("All inputs must be 1D arrays.")
    T = len(years)
    if not all(len(arr) == T for arr in (S, I, R, C, dR, nests)):
        raise ValueError("years, S, I, R, C, dR, nests must share the same length.")

    if T < 3:
        raise ValueError("At least three yearly observations are required to build transition features.")

    # Mid-point states for transitions t -> t+1.
    S_mid = 0.5 * (S[:-1] + S[1:])
    I_mid = 0.5 * (I[:-1] + I[1:])
    R_mid = 0.5 * (R[:-1] + R[1:])
    C_mid = 0.5 * (C[:-1] + C[1:])

    dR_trans = dR[1:T]
    nests_trans = nests[1:T]
    years_trans = years[1:T]

    features: List[np.ndarray] = []
    names: List[str] = []

    _add_feature(features, names, "bias", np.ones_like(S_mid))

    S_ratio = _normalise(S_mid, N)
    I_ratio = _normalise(I_mid, N)
    R_ratio = _normalise(R_mid, N)
    C_ratio = _normalise(C_mid, N)
    dR_ratio = _normalise(dR_trans, N)
    nests_ratio = _normalise(nests_trans, N)

    _add_feature(features, names, "S_ratio", S_ratio)
    _add_feature(features, names, "I_ratio", I_ratio)
    _add_feature(features, names, "R_ratio", R_ratio)
    _add_feature(features, names, "C_ratio", C_ratio)
    _add_feature(features, names, "dR_ratio", dR_ratio)
    _add_feature(features, names, "nests_ratio", nests_ratio)
    _add_feature(features, names, "SI_ratio", S_ratio * I_ratio)
    _add_feature(features, names, "IR_ratio", I_ratio * R_ratio)
    _add_feature(features, names, "SR_ratio", S_ratio * R_ratio)

    _add_feature(features, names, "I_ratio_sq", I_ratio ** 2)
    _add_feature(features, names, "S_ratio_sq", S_ratio ** 2)
    _add_feature(features, names, "R_ratio_sq", R_ratio ** 2)

    # Log-squeezed terms capture saturation behaviour while remaining bounded.
    _add_feature(features, names, "log1p_I_ratio", np.log1p(np.clip(I_ratio, 0, None)))
    _add_feature(features, names, "log1p_S_ratio", np.log1p(np.clip(S_ratio, 0, None)))
    _add_feature(features, names, "log1p_R_ratio", np.log1p(np.clip(R_ratio, 0, None)))
    _add_feature(features, names, "log1p_dR_ratio", np.log1p(np.clip(dR_ratio, 0, None)))

    # Centre the year index to avoid collinearity with the bias term.
    year_c = years_trans - years_trans.mean()
    _add_feature(features, names, "year_centered", year_c / max(np.std(year_c), 1.0))

    if include_exponential:
        _add_feature(features, names, "exp_neg_I_ratio", np.exp(-np.clip(I_ratio, 0, None)))
        _add_feature(features, names, "exp_neg_R_ratio", np.exp(-np.clip(R_ratio, 0, None)))

    if include_trigonometric:
        _add_feature(features, names, "sin_I_ratio", np.sin(I_ratio))
        _add_feature(features, names, "sin_R_ratio", np.sin(R_ratio))

    matrix = np.column_stack(features)

    metadata: Dict[str, np.ndarray] = {
        "S_ratio": S_ratio,
        "I_ratio": I_ratio,
        "R_ratio": R_ratio,
        "C_ratio": C_ratio,
        "dR_ratio": dR_ratio,
        "nests_ratio": nests_ratio,
    }

    return TransitionFeatures(
        matrix=matrix,
        names=names,
        years=years_trans,
        S_mid=S_mid,
        I_mid=I_mid,
        R_mid=R_mid,
        C_mid=C_mid,
        dR=dR_trans,
        nests=nests_trans,
        metadata=metadata,
    )

