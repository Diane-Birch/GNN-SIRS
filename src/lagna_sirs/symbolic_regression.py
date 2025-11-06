"""
Sparse symbolic regression utilities for SIRS parameters.

We fit simple linear models in log-space using a compact feature library.  The
resulting formulas expose β/γ/ω as explicit combinations of interpretable
covariates, following the LaGNA philosophy of separating representation learning
from structuring the dynamical equations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
from sklearn.linear_model import LassoCV, LassoLarsIC, LinearRegression
from sklearn.preprocessing import StandardScaler

from .feature_library import TransitionFeatures


@dataclass
class SymbolicModel:
    """Stores a fitted sparse model ``log θ = intercept + Σ w_i φ_i``."""

    target_name: str
    intercept: float
    coefficients: Dict[str, float]
    feature_order: List[str]
    support: List[str]
    r2_log: float
    residual_std_log: float
    y_mean: float
    solver: str = "lasso_lars_ic"
    transform: str = "exp"

    def as_formula(self) -> str:
        """Pretty-print the symbolic expression."""
        pieces: List[str] = []
        for fname in self.feature_order:
            coef = self.coefficients.get(fname)
            if coef is None:
                continue
            pieces.append(f"{coef:+.4g}·{fname}")
        linear_part = " ".join(pieces) if pieces else "0"
        return f"{self.target_name}(t) = exp({self.intercept:+.4g} {linear_part})"

    def to_dict(self) -> Dict[str, object]:
        return dict(
            target=self.target_name,
            transform=self.transform,
            log_intercept=self.intercept,
            coefficients=self.coefficients,
            support=self.support,
            r2_log=self.r2_log,
            residual_std_log=self.residual_std_log,
            y_mean=self.y_mean,
            formula=self.as_formula(),
        )


def fit_symbolic_model(
    *,
    target_name: str,
    y: Sequence[float],
    features: TransitionFeatures,
    min_std: float = 1e-8,
    max_zero_coef: float = 1e-6,
) -> SymbolicModel:
    """
    Fit a sparse linear model in log-space.

    Parameters
    ----------
    target_name:
        Name of the SIRS parameter (β, γ, ω) used in outputs.
    y:
        Positive target sequence aligned with the transitions.
    features:
        Feature library ``TransitionFeatures`` with ``matrix.shape[0] == len(y)``.
    min_std:
        Threshold to drop near-constant predictors before fitting.
    max_zero_coef:
        Absolute value below which coefficients are treated as zero.
    """

    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim != 1:
        raise ValueError("y must be one-dimensional.")

    if features.matrix.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"Target length {y_arr.shape[0]} does not match feature rows {features.matrix.shape[0]}."
        )

    y_clipped = np.clip(y_arr, 1e-12, None)
    log_y = np.log(y_clipped)

    X = features.matrix
    names = features.names

    stds = X.std(axis=0)
    mask = stds > min_std
    if mask.sum() == 0 or y_arr.size <= 2:
        intercept = float(np.mean(log_y))
        coeffs: Dict[str, float] = {}
        residual = log_y - intercept
        resid_std = float(np.std(residual, ddof=1)) if y_arr.size > 1 else 0.0
        return SymbolicModel(
            target_name=target_name,
            intercept=intercept,
            coefficients=coeffs,
            feature_order=names,
            support=[],
            r2_log=0.0,
            residual_std_log=resid_std,
            y_mean=float(np.mean(y_clipped)),
        )

    X_use = X[:, mask]
    names_use = [names[i] for i, keep in enumerate(mask) if keep]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_use)
    noise_var = float(np.var(log_y, ddof=1)) if y_arr.size > 1 else 0.0
    if noise_var <= 0.0:
        noise_var = 1e-6

    solver_used = "lasso_lars_ic"
    model = LassoLarsIC(criterion="bic", noise_variance=noise_var)
    try:
        model.fit(X_scaled, log_y)
    except ValueError:
        # Fall back to a simple LassoCV (with minimal folds) if BIC-based selection
        # cannot run due to extremely small sample size.
        cv_folds = max(2, min(5, y_arr.size - 1)) if y_arr.size > 2 else 2
        solver_used = "lasso_cv"
        lasso_cv = LassoCV(cv=cv_folds, n_alphas=32, random_state=42)
        lasso_cv.fit(X_scaled, log_y)
        model = lasso_cv
    except Exception:
        # Last resort: ordinary least squares (may be dense).
        solver_used = "ols"
        lin = LinearRegression()
        lin.fit(X_scaled, log_y)
        model = lin

    scales = scaler.scale_.copy()
    scales[scales < min_std] = 1.0

    coef_scaled = getattr(model, "coef_", np.zeros_like(scales))
    intercept_scaled = getattr(model, "intercept_", float(np.mean(log_y)))
    coefs = coef_scaled / scales
    intercept = intercept_scaled - np.sum((scaler.mean_ / scales) * coef_scaled)

    log_pred = intercept + X_use @ coefs

    residual = log_y - log_pred
    denom = np.sum((log_y - log_y.mean()) ** 2)
    r2 = 1.0 - float(np.sum(residual ** 2) / denom) if denom > 1e-10 else 1.0
    resid_std = float(np.std(residual, ddof=1)) if y_arr.size > 1 else 0.0

    coefficients: Dict[str, float] = {}
    support = []
    for name, coef in zip(names_use, coefs):
        if abs(coef) <= max_zero_coef:
            continue
        coefficients[name] = float(coef)
        support.append(name)

    return SymbolicModel(
        target_name=target_name,
        intercept=float(intercept),
        coefficients=coefficients,
        feature_order=names,
        support=support,
        r2_log=float(r2),
        residual_std_log=resid_std,
        y_mean=float(np.mean(y_clipped)),
        solver=solver_used,
    )
