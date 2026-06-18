import numpy as np
import pandas as pd

from src.graph.paysim_graph import build_paysim_graph


def test_build_paysim_graph_unifies_accounts():
    n = 5
    # Accounts: C1, C2, C3, M1. Note C2 is a sender in row 0 and a receiver
    # in row 2 -> it must be ONE shared account node, not two.
    raw_df = pd.DataFrame(
        {
            "nameOrig": ["C1", "C2", "C1", "C3", "C3"],
            "nameDest": ["C2", "M1", "C2", "M1", "C1"],
        }
    )
    feature_df = pd.DataFrame(
        {
            "isFraud": [0, 1, 0, 0, 1],
            "f1": np.arange(n, dtype=float),
            "f2": np.arange(n, dtype=float) * 2,
        },
        index=raw_df.index,
    )
    splits = {
        "train": raw_df.iloc[:3],
        "val": raw_df.iloc[3:4],
        "test": raw_df.iloc[4:5],
    }

    data = build_paysim_graph(
        raw_df=raw_df,
        feature_df=feature_df,
        feature_cols=["f1", "f2"],
        orig_col="nameOrig",
        dest_col="nameDest",
        label_col="isFraud",
        splits=splits,
        add_reverse_edges=True,
    )

    assert data["transaction"].x.shape == (5, 2)
    assert data["transaction"].y.tolist() == [0, 1, 0, 0, 1]
    assert data["transaction"].train_mask.sum().item() == 3
    assert data["transaction"].val_mask.sum().item() == 1
    assert data["transaction"].test_mask.sum().item() == 1

    # 4 unique accounts across both columns: C1, C2, C3, M1
    assert data["account"].x.shape[0] == 4
    # account feature dim = log1p(degree) + mean(feature_cols) = 1 + 2
    assert data["account"].x.shape[1] == 3

    # one sends_from and one sends_to edge per transaction
    assert data["transaction", "sends_from", "account"].edge_index.shape[1] == 5
    assert data["transaction", "sends_to", "account"].edge_index.shape[1] == 5
    assert data["account", "rev_sends_from", "transaction"].edge_index.shape[1] == 5
    assert data["account", "rev_sends_to", "transaction"].edge_index.shape[1] == 5


def test_no_reverse_edges_when_disabled():
    raw_df = pd.DataFrame({"nameOrig": ["C1", "C2"], "nameDest": ["C2", "C1"]})
    feature_df = pd.DataFrame({"isFraud": [0, 1], "f1": [1.0, 2.0]}, index=raw_df.index)

    data = build_paysim_graph(
        raw_df=raw_df,
        feature_df=feature_df,
        feature_cols=["f1"],
        orig_col="nameOrig",
        dest_col="nameDest",
        label_col="isFraud",
        add_reverse_edges=False,
    )

    assert ("account", "rev_sends_from", "transaction") not in data.edge_types
    assert ("account", "rev_sends_to", "transaction") not in data.edge_types
