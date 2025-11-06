#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualise node-level SIRS diffusion outputs produced by Richmond-node-sirs.py.

Generates:
  - diffusion_grid.png: scatter grids coloured by infection probability per year;
  - top_edges.png: top-k transmission edges drawn over the map with node colours;
  - probability_summary.png: average infection probability vs. year.
"""

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise Richmond node SIRS outputs.")
    parser.add_argument("--nodes_csv", type=str, default="node_sirs_out/node_infection_probabilities.csv")
    parser.add_argument("--edges_csv", type=str, default="node_sirs_out/top_transmission_edges.csv")
    parser.add_argument("--out_dir", type=str, default="node_sirs_out/figures")
    parser.add_argument("--years", type=int, nargs="*", default=list(range(2013, 2021)))
    parser.add_argument("--top_k_edges", type=int, default=200, help="Number of edges to plot in the network view.")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="Highlight nodes above this probability.")
    return parser.parse_args()


def ensure_columns(df: pd.DataFrame, years: Iterable[int]) -> List[str]:
    cols = []
    for year in years:
        col = f"prob_{year}"
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing in nodes CSV.")
        cols.append(col)
    return cols


def make_diffusion_grid(df_nodes: pd.DataFrame, years: List[int], out_path: Path):
    prob_cols = ensure_columns(df_nodes, years)
    n_years = len(years)
    n_cols = min(4, n_years)
    n_rows = int(np.ceil(n_years / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols, 4.4 * n_rows), squeeze=False)
    norms = [df_nodes[col].max() for col in prob_cols]
    vmax = max(norms) if norms else 1.0
    vmax = vmax if vmax > 0 else 1.0

    for idx, year in enumerate(years):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]
        scatter = ax.scatter(
            df_nodes["easting"],
            df_nodes["northing"],
            c=df_nodes[f"prob_{year}"],
            s=6,
            cmap="viridis",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(f"{year}")
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        ax.grid(alpha=0.2)
    # hide unused axes
    for j in range(idx + 1, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        axes[row][col].axis("off")

    cbar = fig.colorbar(scatter, ax=axes.ravel().tolist(), shrink=0.9, label="Infection probability")
    cbar.ax.set_title("")
    fig.tight_layout()
    fig.savefig(out_path / "diffusion_grid.png", dpi=220)
    plt.close(fig)


def make_top_edges_plot(
    df_nodes: pd.DataFrame,
    df_edges: pd.DataFrame,
    years: List[int],
    top_k: int,
    threshold: float,
    out_path: Path,
):
    if df_edges.empty:
        return
    df_edges = df_edges.sort_values("transmission_flow", ascending=False).head(top_k)

    coords = df_nodes.set_index("node_idx")[["easting", "northing"]]
    mean_prob_col = f"prob_{years[-1]}"
    vmax = max(df_nodes[f"prob_{year}"].max() for year in years)
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    ax.scatter(
        df_nodes["easting"],
        df_nodes["northing"],
        c=df_nodes[mean_prob_col],
        cmap="plasma",
        s=10,
        vmin=0.0,
        vmax=vmax,
        label="Nodes",
    )

    for _, row in df_edges.iterrows():
        try:
            src_coord = coords.loc[row["src"]].to_numpy()
            dst_coord = coords.loc[row["dst"]].to_numpy()
        except KeyError:
            continue
        color = "tab:orange" if row["transmission_flow"] >= threshold else "tab:gray"
        ax.plot([src_coord[0], dst_coord[0]], [src_coord[1], dst_coord[1]], color=color, alpha=0.4, linewidth=1.0)

    ax.set_title("Top transmission edges")
    ax.set_xlabel("Easting")
    ax.set_ylabel("Northing")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path / "top_edges.png", dpi=240)
    plt.close(fig)


def make_probability_summary(df_nodes: pd.DataFrame, years: List[int], out_path: Path):
    means = [df_nodes[f"prob_{year}"].mean() for year in years]
    medians = [df_nodes[f"prob_{year}"].median() for year in years]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(years, means, marker="o", label="Mean probability")
    ax.plot(years, medians, marker="s", linestyle="--", label="Median probability")
    ax.set_xlabel("Year")
    ax.set_ylabel("Probability")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path / "probability_summary.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_nodes = pd.read_csv(args.nodes_csv)
    df_edges = pd.read_csv(args.edges_csv) if Path(args.edges_csv).exists() else pd.DataFrame()
    if not df_edges.empty:
        df_edges["src"] = df_edges["src"].astype(int)
        df_edges["dst"] = df_edges["dst"].astype(int)

    make_diffusion_grid(df_nodes, args.years, out_dir)
    edge_thresh = df_edges["transmission_flow"].quantile(0.5) if not df_edges.empty else 0.0
    make_top_edges_plot(df_nodes, df_edges, args.years, args.top_k_edges, edge_thresh, out_dir)
    make_probability_summary(df_nodes, args.years, out_dir)
    print(f"[Visualisation] Figures saved to {out_dir.resolve()}")
