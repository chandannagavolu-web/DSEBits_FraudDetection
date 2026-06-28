"""Unit tests for the Elliptic Bitcoin (fraud / GNN) track.

Focus on the track-specific correctness risks: the 3-way label mapping, the
temporal (out-of-time) masks, the AF/LF feature selection with train-only
scaling, and the homogeneous graph assembly.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.elliptic import (
    ID_COL,
    LABEL_COL,
    TIME_COL,
    _feature_column_names,
    _map_class,
    load_edges,
    load_nodes,
    make_temporal_masks,
)
from src.features.elliptic_features import get_feature_columns, scale_features
from src.graph.elliptic_graph import build_elliptic_graph


def make_nodes(n: int = 200, n_feats: int = 165) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = _feature_column_names(n_feats + 2)  # +2 for txId + time_step
    data = {ID_COL: np.arange(1000, 1000 + n), TIME_COL: rng.integers(1, 50, n)}
    for c in cols[2:]:
        data[c] = rng.normal(size=n)
    df = pd.DataFrame(data)
    # ~55% labelled, of which a minority illicit.
    label = np.full(n, -1)
    labelled = rng.random(n) < 0.55
    illicit = labelled & (rng.random(n) < 0.2)
    label[labelled] = 0
    label[illicit] = 1
    df[LABEL_COL] = label
    return df


# ----------------------------------------------------------- label mapping ---
def test_map_class_three_way():
    s = pd.Series(["1", "2", "unknown", "2", "1"])
    assert _map_class(s).tolist() == [1, 0, -1, 0, 1]


def test_feature_column_names_layout():
    names = _feature_column_names(167)  # the real file width
    assert names[0] == ID_COL and names[1] == TIME_COL
    assert names.count(TIME_COL) == 1
    local = [c for c in names if c.startswith("local_")]
    agg = [c for c in names if c.startswith("agg_")]
    # 94 local incl. time step -> 93 local_* ; 72 aggregated.
    assert len(local) == 93
    assert len(agg) == 72


def test_load_nodes_and_edges_roundtrip(tmp_path):
    # header-less features file: txId, time_step, 3 feats
    feats = pd.DataFrame({0: [1, 2, 3], 1: [1, 5, 40], 2: [0.1, 0.2, 0.3], 3: [1, 2, 3], 4: [9, 8, 7]})
    feats.to_csv(tmp_path / "f.csv", header=False, index=False)
    pd.DataFrame({"txId": [1, 2, 3], "class": ["1", "2", "unknown"]}).to_csv(
        tmp_path / "c.csv", index=False
    )
    pd.DataFrame({"txId1": [1, 2], "txId2": [2, 3]}).to_csv(tmp_path / "e.csv", index=False)

    nodes = load_nodes(tmp_path, "f.csv", "c.csv")
    assert nodes[LABEL_COL].tolist() == [1, 0, -1]
    assert list(nodes.columns[:2]) == [ID_COL, TIME_COL]
    edges = load_edges(tmp_path, "e.csv")
    assert edges.shape == (2, 2)


# --------------------------------------------------------- temporal masks -----
def test_temporal_masks_are_ordered_and_label_gated():
    nodes = make_nodes()
    masks = make_temporal_masks(nodes, val_min_step=31, train_max_step=34, test_min_step=35)
    step = nodes[TIME_COL].to_numpy()
    label = nodes[LABEL_COL].to_numpy()

    # train strictly before val before test (out-of-time)
    assert step[masks["train_mask"]].max() < 31
    assert step[masks["val_mask"]].min() >= 31 and step[masks["val_mask"]].max() <= 34
    assert step[masks["test_mask"]].min() >= 35

    # unknown nodes are excluded from every split mask
    for name in ("train_mask", "val_mask", "test_mask"):
        assert (label[masks[name]] != -1).all()

    # masks are disjoint
    overlap = masks["train_mask"] & masks["val_mask"]
    assert not overlap.any()
    assert not (masks["val_mask"] & masks["test_mask"]).any()


def test_temporal_masks_reject_bad_step_order():
    nodes = make_nodes()
    with pytest.raises(ValueError):
        make_temporal_masks(nodes, val_min_step=35, train_max_step=34, test_min_step=30)


# --------------------------------------------------- feature selection --------
def test_feature_set_widths_and_time_step_drop():
    nodes = make_nodes()
    af = get_feature_columns(nodes, "all", n_local_features=94, drop_time_step_feature=False)
    lf = get_feature_columns(nodes, "local", n_local_features=94, drop_time_step_feature=False)
    assert TIME_COL in af and len(af) == 166
    assert len(lf) == 94 and all(not c.startswith("agg_") for c in lf)

    dropped = get_feature_columns(nodes, "all", n_local_features=94, drop_time_step_feature=True)
    assert TIME_COL not in dropped and len(dropped) == 165


def test_scaling_is_fit_on_train_only():
    nodes = make_nodes()
    masks = make_temporal_masks(nodes)
    cols = get_feature_columns(nodes, "all")
    X, scaler = scale_features(nodes, cols, masks["train_mask"], standardize=True)
    # train-node features are standardized (~0 mean / unit std); val/test are not
    # forced to zero mean, proving the scaler saw only train statistics.
    assert np.allclose(X[masks["train_mask"]].mean(axis=0), 0, atol=1e-6)
    assert np.allclose(X[masks["train_mask"]].std(axis=0), 1, atol=1e-5)
    assert scaler is not None


# ------------------------------------------------------- graph assembly -------
def test_build_graph_edges_masks_and_time_step():
    nodes = make_nodes(150)
    masks = make_temporal_masks(nodes)
    cols = get_feature_columns(nodes, "all")
    X, _ = scale_features(nodes, cols, masks["train_mask"])

    # txId-based edges, including one that references a non-existent node (dropped)
    edges = pd.DataFrame({
        "txId1": [nodes[ID_COL].iloc[0], nodes[ID_COL].iloc[1], 999999],
        "txId2": [nodes[ID_COL].iloc[2], nodes[ID_COL].iloc[3], nodes[ID_COL].iloc[0]],
    })
    g = build_elliptic_graph(nodes, edges, X, masks, add_reverse_edges=True)

    n = len(nodes)
    assert g.x.shape == (n, len(cols))
    assert g.edge_index.max().item() < n  # all indices valid
    # 2 valid directed edges, doubled by reverse edges
    assert g.edge_index.shape[1] == 4
    assert g.time_step.shape == (n,)
    # unknown labels (-1) survive on the node tensor for message passing
    assert (g.y == -1).sum().item() == int((nodes[LABEL_COL] == -1).sum())
    for name in ("train_mask", "val_mask", "test_mask", "labeled_mask"):
        assert g[name].dtype == torch.bool
