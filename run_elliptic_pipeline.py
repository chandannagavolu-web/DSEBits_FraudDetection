"""End-to-end Elliptic pipeline: graph -> baselines -> GNN -> temporal analysis.

Assumes the three raw CSVs are under ``data/raw/elliptic/`` (see
``data/README.md``); it does not download anything. Builds the homogeneous
transaction graph, trains the traditional baselines and the homogeneous GNN, and
runs the dark-market temporal-robustness evaluation.

Usage:
    python run_elliptic_pipeline.py
"""

from pathlib import Path

import torch

from src.data.elliptic import load_and_split
from src.evaluation.evaluate_elliptic import run as evaluate_elliptic
from src.graph.elliptic_graph import _feature_columns_and_matrix, build_elliptic_graph
from src.models.baseline.train_baseline_elliptic import run as train_baselines
from src.models.gnn.train_gnn_elliptic import run as train_gnn
from src.utils.io import load_config

cfg = load_config("configs/elliptic.yaml")

# 1. Load nodes + edges, build the temporal masks and the homogeneous graph.
nodes, edges, masks = load_and_split(cfg)
feature_cols, X = _feature_columns_and_matrix(nodes, cfg, masks)
graph = build_elliptic_graph(nodes, edges, X, masks, add_reverse_edges=cfg["gnn"]["add_reverse_edges"])

processed_dir = Path(cfg["paths"]["processed_dir"])
processed_dir.mkdir(parents=True, exist_ok=True)
torch.save(graph, processed_dir / "graph.pt")
print(graph)
print(f"  n_features = {len(feature_cols)} ({cfg['features']['feature_set']})")
for name in ("train_mask", "val_mask", "test_mask"):
    m = masks[name]
    rate = nodes.loc[m, "label"].mean() if m.any() else float("nan")
    print(f"  {name}: {int(m.sum())} nodes, illicit rate = {rate:.4%}")

# 2. Traditional baselines (graph-agnostic) on the same node features.
print("\n=== baselines ===")
train_baselines(cfg)

# 3. Homogeneous GNN.
print("\n=== GNN ===")
train_gnn(cfg)

# 4. Temporal-robustness (dark-market) analysis.
print("\n=== temporal evaluation ===")
evaluate_elliptic(cfg)

print("\nCompleted Elliptic workflow")
