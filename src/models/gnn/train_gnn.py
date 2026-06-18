"""Train and evaluate the heterogeneous GNN on the IEEE-CIS transaction graph.

Loads the ``HeteroData`` graph from ``src/graph/ieee_cis_graph.py``, trains
a binary fraud classifier on ``transaction`` nodes (full-batch, transductive
setting — message passing sees all nodes, the loss only sees
``train_mask``), and reports the shared metrics from
``src/evaluation/metrics.py`` on the val/test masks for comparison against
the baselines in ``src/models/baseline/train_baseline.py``.

Usage:
    python -m src.models.gnn.train_gnn
"""

import gc
from pathlib import Path

import torch
import torch.nn.functional as F

from src.evaluation.metrics import compute_all_metrics
from src.models.gnn.hetero_gnn import HeteroGNN
from src.utils.io import load_config, log_results
from src.utils.seed import set_seed


def run(cfg: dict) -> None:
    set_seed(cfg["seed"])
    # cap BLAS thread pool: each thread holds its own workspace buffers, which
    # adds up on the full ~590K-node graph and was contributing to the memory
    # pressure that led to swap-thrashing on this 8GB machine.
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processed_dir = Path(cfg["paths"]["processed_dir"])
    graph = torch.load(processed_dir / "graph.pt", weights_only=False)
    graph = graph.to(device)

    gnn_cfg = cfg["gnn"]
    in_channels_dict = {node_type: graph[node_type].x.shape[1] for node_type in graph.node_types}

    model = HeteroGNN(
        metadata=graph.metadata(),
        in_channels_dict=in_channels_dict,
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
        model_type=gnn_cfg["model_type"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=gnn_cfg["lr"], weight_decay=gnn_cfg["weight_decay"])

    y = graph["transaction"].y.float()
    train_mask = graph["transaction"].train_mask
    val_mask = graph["transaction"].val_mask
    test_mask = graph["transaction"].test_mask

    n_pos = y[train_mask].sum()
    n_neg = train_mask.sum() - n_pos
    pos_weight = torch.clamp(n_neg / n_pos.clamp(min=1), max=gnn_cfg["pos_weight_clip"]).to(device)

    best_val_pr_auc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, gnn_cfg["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x_dict, graph.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(out[train_mask], y[train_mask], pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            out = model(graph.x_dict, graph.edge_index_dict)
            probs = torch.sigmoid(out)
            val_metrics = compute_all_metrics(y[val_mask].cpu().numpy(), probs[val_mask].cpu().numpy())

        print(f"epoch {epoch:03d}: loss={loss.item():.4f}, val_pr_auc={val_metrics['pr_auc']:.4f}", flush=True)

        if val_metrics["pr_auc"] > best_val_pr_auc:
            best_val_pr_auc = val_metrics["pr_auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= gnn_cfg["patience"]:
                print(f"Early stopping at epoch {epoch} (best val PR-AUC = {best_val_pr_auc:.4f})", flush=True)
                break

        # drop references to this epoch's graph/eval tensors and force
        # collection -- on this memory-constrained machine, deferred GC of
        # the many small per-edge-type intermediate tensors from HeteroConv
        # was letting resident memory grow epoch over epoch into swap.
        del out, loss, probs, val_metrics
        gc.collect()

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(graph.x_dict, graph.edge_index_dict))

    for split_name, mask in [("val", val_mask), ("test", test_mask)]:
        metrics = compute_all_metrics(y[mask].cpu().numpy(), probs[mask].cpu().numpy())
        log_results(
            {
                "experiment": cfg["experiment_name"],
                "model": gnn_cfg["model_type"],
                "resampling": "none",
                "split": split_name,
                "n_features": in_channels_dict["transaction"],
                **metrics,
            },
            cfg["paths"]["results_csv"],
        )
        print(f"[{gnn_cfg['model_type']}] {split_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"{gnn_cfg['model_type']}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved model -> {out_path}")


if __name__ == "__main__":
    config = load_config("configs/ieee_cis.yaml")
    run(config)
