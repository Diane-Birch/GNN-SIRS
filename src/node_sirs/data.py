"""
Utilities for constructing node-level SIRS datasets from RichmondFullData.csv.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import radius_neighbors_graph


@dataclass
class RichmondNodeSeries:
    node_keys: List[Tuple[int, int]]
    coords: np.ndarray
    years: np.ndarray
    counts: np.ndarray
    events: np.ndarray
    total_counts: np.ndarray
    node_index: Dict[Tuple[int, int], int]


def load_richmond_node_series(
    csv_path: str,
    years: Optional[Sequence[int]] = None,
    min_total_count: int = 0,
) -> RichmondNodeSeries:
    """
    Parse RichmondFullData.csv into per-node time series.

    Parameters
    ----------
    csv_path:
        Path to the CSV file (columns Easting, Northing, Year, Nests_Removed).
    years:
        Optional ordered list of years to keep.  If omitted, the sorted unique
        years found in the file are used.
    min_total_count:
        Discard nodes whose total nests_removed across all years is <= this value.
        Use this to prune completely inactive nodes when modelling the diffusion.
    """

    df = pd.read_csv(csv_path)
    if years is None:
        years_arr = np.sort(df["Year"].unique())
    else:
        years_arr = np.array(sorted(set(years)), dtype=np.int32)

    year_to_idx = {int(y): idx for idx, y in enumerate(years_arr)}

    missing_years = sorted(set(df["Year"].unique()) - set(year_to_idx))
    if missing_years:
        raise ValueError(f"Years {missing_years} not covered by provided 'years'.")

    df = df[df["Year"].isin(year_to_idx)].copy()

    node_keys = list({(int(row.Easting), int(row.Northing)) for row in df.itertuples(index=False)})
    node_keys.sort()
    node_index = {key: idx for idx, key in enumerate(node_keys)}
    coords = np.array(node_keys, dtype=np.float32)

    N = len(node_keys)
    T = len(years_arr)
    counts = np.zeros((N, T), dtype=np.float32)

    for row in df.itertuples(index=False):
        key = (int(row.Easting), int(row.Northing))
        idx = node_index[key]
        t = year_to_idx[int(row.Year)]
        counts[idx, t] += float(row.Nests_Removed)

    total_counts = counts.sum(axis=1)
    if min_total_count > 0:
        mask = total_counts > float(min_total_count)
        if mask.sum() == 0:
            raise ValueError("Filtering removed every node. Relax min_total_count.")
        counts = counts[mask]
        coords = coords[mask]
        total_counts = total_counts[mask]
        node_keys = [key for key, keep in zip(node_keys, mask) if keep]
        node_index = {key: idx for idx, key in enumerate(node_keys)}
        N = counts.shape[0]

    events = (counts > 0).astype(np.float32)

    return RichmondNodeSeries(
        node_keys=node_keys,
        coords=coords,
        years=years_arr.astype(np.int32),
        counts=counts,
        events=events,
        total_counts=total_counts,
        node_index=node_index,
    )


def build_radius_graph(
    coords: np.ndarray,
    radius: float,
    include_self: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build undirected edges using a spatial radius (in the same units as coordinates).

    Returns
    -------
    edge_index:
        Array with shape [2, E] containing undirected edges (i->j and j->i).
    edge_weight:
        Placeholder ones array aligned with edge_index (allows weighted variants later).
    """

    if radius <= 0:
        raise ValueError("radius must be positive.")

    graph = radius_neighbors_graph(coords, radius=radius, mode="distance", include_self=include_self)
    coo = graph.tocoo()

    row = coo.row.astype(np.int64)
    col = coo.col.astype(np.int64)

    mask = row != col
    row = row[mask]
    col = col[mask]
    data = np.ones_like(row, dtype=np.float32)

    # ensure undirected symmetry
    row_sym = np.concatenate([row, col], axis=0)
    col_sym = np.concatenate([col, row], axis=0)
    data_sym = np.concatenate([data, data], axis=0)

    edge_index = np.stack([row_sym, col_sym], axis=0)
    return edge_index, data_sym

