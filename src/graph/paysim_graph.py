"""Heterogeneous transaction-graph construction for PaySim fraud detection.

Builds a PyTorch Geometric ``HeteroData`` object with:

- a ``transaction`` node type (one node per transaction, features = the
  scaled feature vector from ``src/features/paysim_features.py``, label =
  ``isFraud``, plus ``train_mask`` / ``val_mask`` / ``test_mask``);
- a single ``account`` node type spanning BOTH ``nameOrig`` and ``nameDest``
  — the same physical account can be a sender in one transaction and a
  receiver in another, so unifying the vocabulary is what lets the GNN
  follow money across hops (fraud rings / layering);
- ``transaction --sends_from--> account`` and
  ``transaction --sends_to--> account`` edges (plus reverses if configured),
  linking each transaction to its originating and destination account.

This differs from ``src/graph/ieee_cis_graph.py``, where each entity column
became its own node type. Here both id columns map into one shared account
index, which is the natural structure for a sender/receiver money-flow graph.

Account node features are ``[log1p(total_degree), mean(transaction features
over all incident transactions)]`` — derived purely from transaction
features, never from the ``isFraud`` label, to avoid target leakage.
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def build_paysim_graph(
    raw_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    feature_cols: List[str],
    orig_col: str,
    dest_col: str,
    label_col: str,
    splits: Dict[str, pd.DataFrame] | None = None,
    add_reverse_edges: bool = True,
) -> HeteroData:
    """Build the heterogeneous transaction/account graph.

    Args:
        raw_df: dataframe carrying the raw ``orig_col`` / ``dest_col`` account
            ids (used only to wire up edges), same index as ``feature_df``.
        feature_df: output of ``PaySimFeaturePipeline.transform(raw_df)``.
        feature_cols: columns of ``feature_df`` used as the transaction node
            feature vector (typically ``pipeline.scaled_feature_cols()``).
        orig_col / dest_col: raw account-id columns in ``raw_df``.
        label_col: column of ``feature_df`` holding the binary fraud label.
        splits: optional dict from :func:`src.data.paysim.make_splits` used to
            build ``train_mask`` / ``val_mask`` / ``test_mask``.
        add_reverse_edges: also add ``account --rev_*--> transaction`` edges
            for bidirectional message passing.
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

    # Shared account vocabulary across both endpoints.
    orig = raw_df[orig_col].astype(str)
    dest = raw_df[dest_col].astype(str)
    all_accounts = pd.unique(pd.concat([orig, dest], ignore_index=True))
    acct_to_idx = {a: i for i, a in enumerate(all_accounts)}
    n_accounts = len(all_accounts)

    tx_pos = np.arange(n_tx, dtype=np.int64)
    orig_idx = orig.map(acct_to_idx).to_numpy(dtype=np.int64)
    dest_idx = dest.map(acct_to_idx).to_numpy(dtype=np.int64)

    data["transaction", "sends_from", "account"].edge_index = torch.from_numpy(
        np.vstack([tx_pos, orig_idx])
    )
    data["transaction", "sends_to", "account"].edge_index = torch.from_numpy(
        np.vstack([tx_pos, dest_idx])
    )
    if add_reverse_edges:
        data["account", "rev_sends_from", "transaction"].edge_index = torch.from_numpy(
            np.vstack([orig_idx, tx_pos])
        )
        data["account", "rev_sends_to", "transaction"].edge_index = torch.from_numpy(
            np.vstack([dest_idx, tx_pos])
        )

    # Account features: degree (as orig + as dest) and the mean transaction
    # feature vector over all incident transactions.
    degree = np.zeros(n_accounts, dtype=np.float32)
    feat_sum = np.zeros((n_accounts, X.shape[1]), dtype=np.float32)
    for idx in (orig_idx, dest_idx):
        np.add.at(degree, idx, 1.0)
        np.add.at(feat_sum, idx, X)

    feat_mean = feat_sum / np.maximum(degree, 1.0)[:, None]
    account_features = np.concatenate([np.log1p(degree)[:, None], feat_mean], axis=1)
    data["account"].x = torch.from_numpy(account_features.astype(np.float32))

    return data


if __name__ == "__main__":
    from src.data.paysim import load_and_split
    from src.features.paysim_features import PaySimFeaturePipeline
    from src.utils.io import load_config

    cfg = load_config("configs/paysim.yaml")
    processed_dir = Path(cfg["paths"]["processed_dir"])

    df, splits = load_and_split(cfg)
    # only split membership (the index) is needed for the masks
    splits = {name: split_df[[]] for name, split_df in splits.items()}

    pipeline = PaySimFeaturePipeline.load(processed_dir / "feature_pipeline.pkl")
    # skip the raw/tree view and use float32 -- the graph only consumes
    # scaled_feature_cols(), and this halves peak memory on the large graph.
    feature_df = pipeline.transform(df, include_raw=False)

    graph_cfg = cfg["graph"]
    orig_col, dest_col = graph_cfg["orig_col"], graph_cfg["dest_col"]
    raw_ids = df[[orig_col, dest_col]]

    graph = build_paysim_graph(
        raw_df=raw_ids,
        feature_df=feature_df,
        feature_cols=pipeline.scaled_feature_cols(),
        orig_col=orig_col,
        dest_col=dest_col,
        label_col=cfg["features"]["label_col"],
        splits=splits,
        add_reverse_edges=graph_cfg["add_reverse_edges"],
    )

    out_path = processed_dir / "graph.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graph, out_path)

    print(graph)
    print(f"Saved heterogeneous graph -> {out_path}")
