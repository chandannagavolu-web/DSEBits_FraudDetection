"""Homogeneous transaction-graph construction for the Elliptic Bitcoin Dataset.

Builds a PyTorch Geometric ``Data`` object — a *single* node type (one node per
Bitcoin transaction), which is the key structural difference from the
heterogeneous IEEE-CIS / PaySim builders. The edges are the dataset's given
directed payment flows; no entities or shared-id linking is constructed here.

Node features are the (optionally standardized) feature matrix from
``src/features/elliptic_features.py``. Labels are illicit=1 / licit=0 /
unknown=-1; the train/val/test masks cover only labelled nodes, while unknown
nodes still participate in message passing.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from src.data.elliptic import ID_COL, LABEL_COL, TIME_COL


def build_elliptic_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    X: np.ndarray,
    masks: Dict[str, np.ndarray],
    add_reverse_edges: bool = True,
) -> Data:
    """Assemble the homogeneous transaction graph.

    Args:
        nodes: node dataframe (row position == node index) carrying ``txId`` and
            ``label``; ``X`` must be row-aligned with it.
        edges: directed edge list with columns ``txId1`` / ``txId2``.
        X: node feature matrix, shape ``(n_nodes, n_features)``, float32.
        masks: boolean node masks from
            :func:`src.data.elliptic.make_temporal_masks`.
        add_reverse_edges: also add the reverse of each directed edge so message
            passing is bidirectional (Bitcoin flows are directed).
    """
    id_to_idx = {tx: i for i, tx in enumerate(nodes[ID_COL].to_numpy())}

    src = edges["txId1"].map(id_to_idx).to_numpy()
    dst = edges["txId2"].map(id_to_idx).to_numpy()
    # Drop any edge whose endpoint isn't in the node table (shouldn't happen on
    # the official files, but keeps synthetic / trimmed inputs safe).
    valid = ~(pd.isna(src) | pd.isna(dst))
    src = src[valid].astype(np.int64)
    dst = dst[valid].astype(np.int64)

    if add_reverse_edges:
        edge_index = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    else:
        edge_index = np.vstack([src, dst])

    data = Data(
        x=torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)),
        edge_index=torch.from_numpy(np.ascontiguousarray(edge_index)),
        y=torch.from_numpy(nodes[LABEL_COL].to_numpy(dtype=np.int64)),
    )
    # Keep the raw time step per node so the evaluation can break performance
    # down by time (the "dark-market shutdown" analysis); it is NOT used as a
    # model input here.
    data.time_step = torch.from_numpy(nodes[TIME_COL].to_numpy(dtype=np.int64))
    for name, mask in masks.items():
        data[name] = torch.from_numpy(np.ascontiguousarray(mask, dtype=bool))
    return data


def _feature_columns_and_matrix(nodes: pd.DataFrame, cfg: dict, masks: Dict[str, np.ndarray]):
    from src.features.elliptic_features import get_feature_columns, scale_features

    feat_cfg = cfg["features"]
    feature_cols: List[str] = get_feature_columns(
        nodes,
        feature_set=feat_cfg["feature_set"],
        n_local_features=feat_cfg["n_local_features"],
        drop_time_step_feature=feat_cfg["drop_time_step_feature"],
    )
    X, _ = scale_features(
        nodes,
        feature_cols,
        masks["train_mask"],
        standardize=feat_cfg["standardize"],
    )
    return feature_cols, X


if __name__ == "__main__":
    from src.data.elliptic import load_and_split
    from src.utils.io import load_config

    cfg = load_config("configs/elliptic.yaml")
    nodes, edges, masks = load_and_split(cfg)

    feature_cols, X = _feature_columns_and_matrix(nodes, cfg, masks)
    graph = build_elliptic_graph(
        nodes,
        edges,
        X,
        masks,
        add_reverse_edges=cfg["gnn"]["add_reverse_edges"],
    )

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_path = processed_dir / "graph.pt"
    torch.save(graph, out_path)

    print(graph)
    print(f"  n_features = {len(feature_cols)} ({cfg['features']['feature_set']})")
    print(f"Saved homogeneous graph -> {out_path}")
