"""Loading and temporal splitting for the Elliptic Bitcoin Dataset
(fraud / systemic-risk track, GNN node classification).

The dataset ships as three CSVs (see ``data/README.md``):

- ``elliptic_txs_features.csv`` — **no header**, 167 columns: ``txId``,
  ``time_step`` (1..49), then 165 anonymised features. With the time step the
  feature block is 166 wide: the first 94 (time step + 93) are *local*
  transaction features, the remaining 72 are *aggregated* one-hop neighbourhood
  roll-ups (Weber et al. 2019).
- ``elliptic_txs_classes.csv`` — header ``txId,class`` with values ``1``
  (illicit), ``2`` (licit) and ``unknown``.
- ``elliptic_txs_edgelist.csv`` — header ``txId1,txId2``, 234K directed payment
  edges.

Unlike the IEEE-CIS / PaySim fraud tracks there is **no feature engineering and
no entity-graph construction** — the graph is given. The two things this module
owns are (1) mapping the 3-way class column onto the binary fraud convention
(illicit = 1, licit = 0, unknown = -1) and (2) the **temporal** train/val/test
split over the 49 time steps, which replaces the random stratified split used by
the other tracks.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.io import load_config
from src.utils.seed import set_seed

ID_COL = "txId"
TIME_COL = "time_step"
LABEL_COL = "label"

# Class-column encoding in elliptic_txs_classes.csv -> binary fraud label.
# illicit = 1 (positive/event), licit = 0, everything else (unknown) = -1.
ILLICIT = 1
LICIT = 0
UNKNOWN = -1


def _feature_column_names(n_cols: int) -> list[str]:
    """Column names for the header-less features CSV.

    Layout: ``txId``, ``time_step``, then ``local_1..local_93`` (the remaining
    local features) and ``agg_1..agg_72`` (aggregated neighbourhood features).
    The split between local and aggregated is positional (94 local incl. the
    time step, 72 aggregated) per the Elliptic benchmark.
    """
    n_feats = n_cols - 2  # drop txId + time_step
    # 94 local features include the time step, so 93 remain — but clamp to the
    # columns actually present so trimmed / synthetic files don't over-name.
    n_local_rest = min(94 - 1, n_feats)
    names = [ID_COL, TIME_COL]
    names += [f"local_{i}" for i in range(1, n_local_rest + 1)]
    names += [f"agg_{i}" for i in range(1, n_feats - n_local_rest + 1)]
    return names


def _map_class(series: pd.Series) -> pd.Series:
    """Map the raw ``class`` column onto illicit=1 / licit=0 / unknown=-1.

    The file codes illicit as ``1`` and licit as ``2`` (strings); ``unknown``
    and any missing join stay -1.
    """
    text = series.astype(str).str.strip().str.lower()
    mapped = pd.Series(UNKNOWN, index=series.index, dtype=int)
    mapped[text == "1"] = ILLICIT
    mapped[text == "2"] = LICIT
    return mapped


def load_nodes(
    raw_dir: str | Path,
    features_file: str = "elliptic_txs_features.csv",
    classes_file: str = "elliptic_txs_classes.csv",
) -> pd.DataFrame:
    """Load node features joined with the (binary-mapped) class label.

    Returns a dataframe indexed by row position with columns ``txId``,
    ``time_step``, the feature columns, and ``label`` (1/0/-1).
    """
    raw_dir = Path(raw_dir)

    feats = pd.read_csv(raw_dir / features_file, header=None)
    feats.columns = _feature_column_names(feats.shape[1])
    feats[ID_COL] = feats[ID_COL].astype("int64")

    classes = pd.read_csv(raw_dir / classes_file)
    classes.columns = [c.strip() for c in classes.columns]
    classes[ID_COL] = classes[ID_COL].astype("int64")
    classes[LABEL_COL] = _map_class(classes["class"])

    df = feats.merge(classes[[ID_COL, LABEL_COL]], on=ID_COL, how="left")
    df[LABEL_COL] = df[LABEL_COL].fillna(UNKNOWN).astype(int)
    return df.reset_index(drop=True)


def load_edges(
    raw_dir: str | Path,
    edgelist_file: str = "elliptic_txs_edgelist.csv",
) -> pd.DataFrame:
    """Load the directed edge list (``txId1 -> txId2``)."""
    raw_dir = Path(raw_dir)
    edges = pd.read_csv(raw_dir / edgelist_file)
    edges.columns = [c.strip() for c in edges.columns]
    return edges.astype("int64")


def make_temporal_masks(
    nodes: pd.DataFrame,
    val_min_step: int = 31,
    train_max_step: int = 34,
    test_min_step: int = 35,
) -> Dict[str, np.ndarray]:
    """Boolean node masks for the temporal (out-of-time) split.

    A node is only eligible for a labelled split if its label is known
    (illicit/licit); unknown nodes are excluded from every mask but remain in
    the graph for message passing.

      train: time_step < val_min_step
      val:   val_min_step <= time_step <= train_max_step
      test:  time_step >= test_min_step
    """
    if not (val_min_step <= train_max_step < test_min_step):
        raise ValueError(
            "Expect val_min_step <= train_max_step < test_min_step, got "
            f"{val_min_step}, {train_max_step}, {test_min_step}"
        )

    step = nodes[TIME_COL].to_numpy()
    labeled = nodes[LABEL_COL].to_numpy() != UNKNOWN

    train = labeled & (step < val_min_step)
    val = labeled & (step >= val_min_step) & (step <= train_max_step)
    test = labeled & (step >= test_min_step)
    return {
        "train_mask": train,
        "val_mask": val,
        "test_mask": test,
        "labeled_mask": labeled,
    }


def load_and_split(config: dict) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, np.ndarray]]:
    """Load nodes + edges and build the temporal masks from the config."""
    set_seed(config["seed"])

    paths = config["paths"]
    nodes = load_nodes(paths["raw_dir"], paths["features_file"], paths["classes_file"])
    edges = load_edges(paths["raw_dir"], paths["edgelist_file"])

    split_cfg = config["split"]
    masks = make_temporal_masks(
        nodes,
        val_min_step=split_cfg["val_min_step"],
        train_max_step=split_cfg["train_max_step"],
        test_min_step=split_cfg["test_min_step"],
    )
    return nodes, edges, masks


if __name__ == "__main__":
    cfg = load_config("configs/elliptic.yaml")
    nodes, edges, masks = load_and_split(cfg)

    n_illicit = int((nodes[LABEL_COL] == ILLICIT).sum())
    n_licit = int((nodes[LABEL_COL] == LICIT).sum())
    n_unknown = int((nodes[LABEL_COL] == UNKNOWN).sum())
    print(
        f"Loaded {len(nodes)} nodes ({nodes[TIME_COL].nunique()} time steps), "
        f"{len(edges)} edges"
    )
    print(f"  illicit={n_illicit}  licit={n_licit}  unknown={n_unknown}")
    for name in ("train_mask", "val_mask", "test_mask"):
        m = masks[name]
        rate = nodes.loc[m, LABEL_COL].mean() if m.any() else float("nan")
        print(f"  {name}: {int(m.sum())} nodes, illicit rate = {rate:.4%}")
