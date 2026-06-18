import numpy as np
import pandas as pd

from src.graph.ieee_cis_graph import build_hetero_graph


def test_build_hetero_graph_basic():
    n = 6
    raw_df = pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "card1": [1, 1, 2, 2, 3, np.nan],
            "addr1": [10, 10, 10, 20, np.nan, np.nan],
        }
    )
    feature_df = pd.DataFrame(
        {
            "isFraud": [0, 1, 0, 0, 1, 0],
            "f1": np.arange(n, dtype=float),
            "f2": np.arange(n, dtype=float) * 2,
        },
        index=raw_df.index,
    )

    splits = {
        "train": raw_df.iloc[:4],
        "val": raw_df.iloc[4:5],
        "test": raw_df.iloc[5:6],
    }

    data = build_hetero_graph(
        raw_df=raw_df,
        feature_df=feature_df,
        feature_cols=["f1", "f2"],
        entity_cols=["card1", "addr1"],
        label_col="isFraud",
        splits=splits,
        add_reverse_edges=True,
    )

    assert data["transaction"].x.shape == (6, 2)
    assert data["transaction"].y.tolist() == [0, 1, 0, 0, 1, 0]
    assert data["transaction"].train_mask.sum().item() == 4
    assert data["transaction"].val_mask.sum().item() == 1
    assert data["transaction"].test_mask.sum().item() == 1

    # card1 has 3 unique non-null values (1, 2, 3)
    assert data["card1"].x.shape[0] == 3
    # addr1 has 2 unique non-null values (10, 20)
    assert data["addr1"].x.shape[0] == 2

    # 5 transactions have a non-null card1
    assert data["transaction", "linked_to", "card1"].edge_index.shape[1] == 5
    assert data["card1", "rev_linked_to", "transaction"].edge_index.shape[1] == 5

    # 4 transactions have a non-null addr1
    assert data["transaction", "linked_to", "addr1"].edge_index.shape[1] == 4
    assert data["addr1", "rev_linked_to", "transaction"].edge_index.shape[1] == 4

    # entity feature dim = log1p(degree) + mean(feature_cols) = 1 + 2
    assert data["card1"].x.shape[1] == 3
    assert data["addr1"].x.shape[1] == 3
