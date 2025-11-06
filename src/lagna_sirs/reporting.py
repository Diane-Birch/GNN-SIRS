"""
Export utilities for symbolic SIRS parameter formulas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence
import pandas as pd

from .feature_library import TransitionFeatures
from .symbolic_regression import SymbolicModel


def _format_model_block(model: SymbolicModel) -> str:
    lines = [
        f"{model.target_name}(t):",
        f"  formula : {model.as_formula()}",
        f"  support : {', '.join(model.support) if model.support else '(bias only)'}",
        f"  r2_log  : {model.r2_log:.3f}",
        f"  σ_resid : {model.residual_std_log:.3f}",
    ]
    return "\n".join(lines)


def export_symbolic_summary(
    models: Sequence[SymbolicModel],
    features: TransitionFeatures,
    out_dir: Path | str,
) -> None:
    """
    Persist symbolic regression outputs (JSON, text, CSV diagnostics).
    """

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_payload = [model.to_dict() for model in models]
    with (out_path / "symbolic_formulas.json").open("w", encoding="utf-8") as f:
        json.dump(json_payload, f, ensure_ascii=False, indent=2)

    text_blocks = [_format_model_block(model) for model in models]
    header = (
        "Interpretable parameter formulas derived from LaGNA-inspired sparse regression.\n"
        "Each formula encodes the log-parameter as a linear combination of engineered features.\n"
    )
    with (out_path / "symbolic_formulas.txt").open("w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n\n".join(text_blocks))
        f.write("\n")

    # Export the transition-level feature table to ease inspection / manual tweaking.
    feature_table = {"year": features.years}
    for idx, name in enumerate(features.names):
        feature_table[name] = features.matrix[:, idx]
    df_features = pd.DataFrame(feature_table)
    df_features.to_csv(out_path / "symbolic_features.csv", index=False, float_format="%.6g")

    # Contextual mid-point states for the transitions.
    df_context = pd.DataFrame(
        {
            "year": features.years,
            "S_mid": features.S_mid,
            "I_mid": features.I_mid,
            "R_mid": features.R_mid,
            "C_mid": features.C_mid,
            "dR": features.dR,
            "nests": features.nests,
        }
    )
    df_context.to_csv(out_path / "symbolic_context.csv", index=False, float_format="%.6g")
