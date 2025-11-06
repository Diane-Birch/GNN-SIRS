#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train a node-level SIRS diffusion model for Richmond Park trees.

This script consumes RichmondFullData.csv, builds a spatial radius graph, and
fits a neural SIRS model that estimates the infection probability for each tree
across the yearly timeline (2013-2020).  The outputs include per-node infection
probabilities, inferred first infection times, and ranked edge flows describing
the most likely transmission paths.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from src.node_sirs import load_richmond_node_series, train_node_sirs


def build_outputs(out_dir: Path, training_out, threshold: float, radius: float, epochs: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = training_out.summary
    event_prob = summary.event_prob.detach().cpu().numpy()  # [T, N]
    infection_prob = summary.infection_prob.detach().cpu().numpy()
    new_infections = summary.new_infections.detach().cpu().numpy()
    years = training_out.years
    coords = training_out.coords.detach().cpu().numpy()
    events = training_out.events.detach().cpu().numpy()  # [T, N]

    node_records = []
    for idx, (x, y) in enumerate(coords):
        probs = {f"prob_{year}": float(event_prob[t, idx]) for t, year in enumerate(years)}
        inf_probs = {f"infection_{year}": float(infection_prob[t, idx]) for t, year in enumerate(years)}
        obs_flags = {f"observed_{year}": float(events[t, idx]) for t, year in enumerate(years)}
        node_records.append(
            dict(
                node_idx=int(idx),
                easting=float(x),
                northing=float(y),
                first_prob_year=_first_above(event_prob[:, idx], years, threshold),
                first_obs_year=_first_above(events[:, idx], years, 0.5),
                total_observed=float(events[:, idx].sum()),
                **probs,
                **inf_probs,
                **obs_flags,
            )
        )
    pd.DataFrame(node_records).to_csv(out_dir / "node_infection_probabilities.csv", index=False, float_format="%.6g")

    Path(out_dir / "loss_history.json").write_text(
        json.dumps(training_out.loss_history.tolist(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    meta = dict(
        gamma=float(training_out.model.gamma.detach().cpu()),
        omega=float(training_out.model.omega.detach().cpu()),
        radius=float(radius),
        threshold=float(threshold),
        epochs=int(epochs),
    )
    Path(out_dir / "model_metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    edge_index = training_out.edge_index.detach().cpu().numpy()
    edge_rows = []
    I = summary.I.detach().cpu().numpy()  # [T, N]
    for src, dst in zip(edge_index[0], edge_index[1]):
        flow = float(np.sum(I[:, src] * infection_prob[:, dst]))
        src_prob_year = _first_above(event_prob[:, src], years, threshold)
        dst_prob_year = _first_above(event_prob[:, dst], years, threshold)
        edge_rows.append(
            dict(
                src=int(src),
                dst=int(dst),
                src_first_prob_year=src_prob_year,
                dst_first_prob_year=dst_prob_year,
                transmission_flow=flow,
            )
        )
    df_edges = pd.DataFrame(edge_rows)
    df_edges.sort_values("transmission_flow", ascending=False).head(500).to_csv(
        out_dir / "top_transmission_edges.csv", index=False, float_format="%.6g"
    )

    torch.save(
        dict(
            model_state=training_out.model.state_dict(),
            coords=coords,
            edge_index=edge_index,
            years=years,
        ),
        out_dir / "node_sirs_model.pt",
    )


def _first_above(series: np.ndarray, years: np.ndarray, threshold: float) -> int:
    mask = series >= threshold
    if not np.any(mask):
        return None
    first_idx = np.argmax(mask)
    return int(years[first_idx])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Node-level SIRS diffusion for Richmond Park.")
    parser.add_argument("--csv_path", type=str, default="RichmondFullData.CSV")
    parser.add_argument("--out_dir", type=str, default="node_sirs_out")
    parser.add_argument("--radius", type=float, default=120.0, help="Neighbourhood radius (same units as coordinates).")
    parser.add_argument("--min_total_count", type=int, default=0, help="Prune nodes with <= this total nests_removed.")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for first infection year.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)

    dataset = load_richmond_node_series(
        args.csv_path,
        min_total_count=args.min_total_count,
    )

    training_out = train_node_sirs(
        dataset,
        radius=args.radius,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    build_outputs(out_dir, training_out, threshold=args.threshold, radius=args.radius, epochs=args.epochs)
    print(f"[Done] Outputs written to {out_dir.resolve()}")
