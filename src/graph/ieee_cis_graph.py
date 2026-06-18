"""Heterogeneous transaction-graph construction for IEEE-CIS Fraud Detection.

Builds a PyTorch Geometric ``HeteroData`` object with:

- a ``transaction`` node type (one node per transaction, features = the
  scaled feature vector from ``src/features/ieee_cis_features.py``,
  label = ``isFraud``, plus ``train_mask`` / ``val_mask`` / ``test_mask``);
- one node type per "entity" column in ``configs/ieee_cis.yaml``
  (``card1``, ``addr1``, ``P_emaildomain``, ``DeviceInfo`` by default) —
  shared identifiers that link transactions together;
- ``transaction --linked_to--> entity`` edges (and the reverse, if
  configured) connecting a transaction to each entity value it shares.
  Missing entity values are skipped so they don't form a giant hub node.

Entity node features are ``[log1p(degree), mean(transaction features over
linked transactions)]`` — derived purely from transaction features, never
from the ``isFraud`` label, to avoid target leakage.

See ``data/README.md`` for the full rationale.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from src.data.ieee_cis import load_and_split
from src.features.ieee_cis_features import IEEECISFeaturePipeline
from src.utils.io import load_config


def build_hetero_graph(
    raw_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    entity_cols: List[str],
    label_col: str,
    splits: Dict[str, pd.DataFrame] | None = None,
    add_reverse_edges: bool = True,
) -> HeteroData:
    """Build the heterogeneous transaction graph.

    Args:
        raw_df: merged transaction+identity dataframe (raw entity values,
            used only to determine which transactions share an entity).
        feature_df: output of ``IEEECISFeaturePipeline.transform(raw_df)``,
            same index as ``raw_df``.
        feature_cols: columns of ``feature_df`` to use as the transaction
            node feature vector (typically ``pipeline.scaled_feature_cols()``).
        entity_cols: columns of ``raw_df`` to turn into entity node types.
        label_col: column of ``feature_df`` holding the binary fraud label.
        splits: optional dict from :func:`src.data.ieee_cis.make_splits`
            (DataFrames sharing ``raw_df``'s index) used to build
            ``train_mask`` / ``val_mask`` / ``test_mask`` on the
            ``transaction`` nodes.
        add_reverse_edges: also add ``entity --rev_linked_to--> transaction``
            edges for bidirectional message passing.
    """
    feature_df = feature_df.loc[raw_df.index]

    X = feature_df[feature_cols].to_numpy(dtype=np.float32)
    n_tx = X.shape[0]

    data = HeteroData()
    data["transaction"].x = torch.from_numpy(X)

    if label_col in feature_df.columns:
        y = feature_df[label_col].to_numpy(dtype=np.int64)
        data["transaction"].y = torch.from_numpy(y)

    if splits is not None:
        index_pos = {idx: pos for pos, idx in enumerate(raw_df.index)}
        for split_name, split_df in splits.items():
            mask = torch.zeros(n_tx, dtype=torch.bool)
            positions = [index_pos[i] for i in split_df.index]
            mask[positions] = True
            data["transaction"][f"{split_name}_mask"] = mask

    for entity_col in entity_cols:
        values = raw_df[entity_col]
        valid = values.notna().to_numpy()

        unique_vals = pd.unique(values[valid])
        val_to_idx = {v: i for i, v in enumerate(unique_vals)}
        n_entities = len(unique_vals)

        tx_idx = np.where(valid)[0]
        ent_idx = values[valid].map(val_to_idx).to_numpy(dtype=np.int64)

        edge_index = torch.from_numpy(np.vstack([tx_idx, ent_idx]).astype(np.int64))
        data["transaction", "linked_to", entity_col].edge_index = edge_index

        if add_reverse_edges:
            rev_edge_index = torch.from_numpy(np.vstack([ent_idx, tx_idx]).astype(np.int64))
            data[entity_col, "rev_linked_to", "transaction"].edge_index = rev_edge_index

        degree = np.zeros(n_entities, dtype=np.float32)
        feat_sum = np.zeros((n_entities, X.shape[1]), dtype=np.float32)
        np.add.at(degree, ent_idx, 1.0)
        np.add.at(feat_sum, ent_idx, X[tx_idx])

        feat_mean = feat_sum / np.maximum(degree, 1.0)[:, None]
        entity_features = np.concatenate([np.log1p(degree)[:, None], feat_mean], axis=1)
        data[entity_col].x = torch.from_numpy(entity_features.astype(np.float32))

    return data


if __name__ == "__main__":
    cfg = load_config("configs/ieee_cis.yaml")
    processed_dir = Path(cfg["paths"]["processed_dir"])

    df, splits = load_and_split(cfg)
    # only split membership (the index) is needed for train/val/test masks,
    # so drop the duplicated feature columns each split otherwise carries
    splits = {name: split_df[[]] for name, split_df in splits.items()}

    pipeline = IEEECISFeaturePipeline.load(processed_dir / "feature_pipeline.pkl")
    # skip the raw/tree view and use float32 for the scaled view -- the graph
    # only consumes scaled_feature_cols(), and this dataset is large enough
    # (~590K rows) that the full dual-view transform risks an OOM
    feature_df = pipeline.transform(df, include_raw=False)

    entity_cols = cfg["graph"]["entity_cols"]
    df = df[entity_cols]

    graph = build_hetero_graph(
        raw_df=df,
        feature_df=feature_df,
        feature_cols=pipeline.scaled_feature_cols(),
        entity_cols=entity_cols,
        label_col=cfg["features"]["label_col"],
        splits=splits,
        add_reverse_edges=cfg["graph"]["add_reverse_edges"],
    )

    out_path = processed_dir / "graph.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, out_path)

    print(graph)
    print(f"Saved heterogeneous graph -> {out_path}")
