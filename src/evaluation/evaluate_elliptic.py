"""Evaluation & temporal-robustness analysis for the Elliptic models.

The training scripts already log overall val/test metrics to
``reports/results_elliptic.csv``. This module adds the analysis that is specific
to Elliptic's *temporal* structure and is the most-cited finding on the dataset:
the **"dark-market shutdown"**. Around time step 43 a major darknet marketplace
was closed, and every model's ability to catch illicit transactions collapses on
the later time steps — a vivid illustration that a fraud model trained on the
past can degrade sharply when the underlying behaviour shifts (concept drift).

It reloads the saved baselines (Logistic Regression / Random Forest / XGBoost)
and the GNN, scores the labelled nodes, and produces:

- per-time-step illicit-class F1 for every model
  -> reports/elliptic/per_timestep_f1.csv
- a combined illicit-class summary on the test span
  -> reports/elliptic/elliptic_illicit_summary.csv
- the dark-market F1-over-time plot
  -> reports/figures/elliptic_dark_market_f1.png

Run after training the baselines and the GNN:
    python -m src.evaluation.evaluate_elliptic
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score

from src.evaluation.metrics import compute_all_metrics
from src.models.gnn.homo_gnn import HomoGNN
from src.utils.io import load_config
from src.utils.seed import set_seed

# The market shutdown that splits Elliptic into an "easy" and a "hard" regime.
DARK_MARKET_STEP = 43


def _load_baseline(model_name: str, model_dir: Path):
    path_json = model_dir / f"{model_name}.json"
    path_pkl = model_dir / f"{model_name}.pkl"
    if model_name == "xgboost" and path_json.exists():
        clf = xgb.XGBClassifier()
        clf.load_model(path_json)
        return clf
    if path_pkl.exists():
        with open(path_pkl, "rb") as f:
            return pickle.load(f)
    return None


def _score_all_models(cfg: dict, graph) -> Dict[str, np.ndarray]:
    """Return {model_name: prob_illicit per node} for every saved model found."""
    model_dir = Path(cfg["paths"]["model_dir"])
    X = graph.x.numpy()
    scores: Dict[str, np.ndarray] = {}

    for model_name in cfg["baseline"]["models"]:
        model = _load_baseline(model_name, model_dir)
        if model is not None:
            scores[model_name] = model.predict_proba(X)[:, 1]

    gnn_type = cfg["gnn"]["model_type"]
    gnn_path = model_dir / f"{gnn_type}.pt"
    if gnn_path.exists():
        model = HomoGNN(
            in_channels=graph.x.shape[1],
            hidden_dim=cfg["gnn"]["hidden_dim"],
            num_layers=cfg["gnn"]["num_layers"],
            dropout=cfg["gnn"]["dropout"],
            model_type=gnn_type,
        )
        model.load_state_dict(torch.load(gnn_path, weights_only=True))
        model.eval()
        with torch.no_grad():
            scores[gnn_type] = torch.sigmoid(model(graph.x, graph.edge_index)).numpy()
    return scores


def run(cfg: dict, threshold: float = 0.5) -> None:
    set_seed(cfg["seed"])

    graph = torch.load(Path(cfg["paths"]["processed_dir"]) / "graph.pt", weights_only=False)
    y = graph.y.numpy()
    time_step = graph.time_step.numpy()
    test_mask = graph.test_mask.numpy()
    labeled = graph.labeled_mask.numpy()

    scores = _score_all_models(cfg, graph)
    if not scores:
        raise FileNotFoundError("No trained models found — run the baselines / GNN first.")

    out_dir = Path(cfg["paths"]["results_csv"]).parent / "elliptic"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- per-time-step illicit F1 (the dark-market analysis) -----------------
    steps = sorted(np.unique(time_step[labeled]))
    rows = []
    for t in steps:
        node_sel = labeled & (time_step == t)
        row = {"time_step": int(t), "n_labeled": int(node_sel.sum()), "n_illicit": int(y[node_sel].sum())}
        for name, prob in scores.items():
            yt, yp = y[node_sel], (prob[node_sel] >= threshold).astype(int)
            # F1 is undefined (0/0) when a step has no illicit cases; report NaN.
            row[name] = f1_score(yt, yp, zero_division=0) if yt.sum() > 0 else np.nan
        rows.append(row)
    per_step = pd.DataFrame(rows)
    per_step_path = out_dir / "per_timestep_f1.csv"
    per_step.to_csv(per_step_path, index=False)

    # ---- illicit-class summary on the test span ------------------------------
    summary_rows = []
    for name, prob in scores.items():
        yt = y[test_mask]
        yp = (prob[test_mask] >= threshold).astype(int)
        m = compute_all_metrics(yt, prob[test_mask])
        summary_rows.append({
            "model": name,
            "illicit_f1": f1_score(yt, yp, zero_division=0),
            "illicit_precision": precision_score(yt, yp, zero_division=0),
            "illicit_recall": recall_score(yt, yp, zero_division=0),
            "pr_auc": m["pr_auc"], "roc_auc": m["roc_auc"], "mcc": m["mcc"],
        })
    summary = pd.DataFrame(summary_rows).sort_values("illicit_f1", ascending=False)
    summary_path = out_dir / "elliptic_illicit_summary.csv"
    summary.to_csv(summary_path, index=False)

    _plot_dark_market(per_step, list(scores), fig_dir / "elliptic_dark_market_f1.png")

    print(f"Saved per-time-step F1 -> {per_step_path}")
    print(f"Saved illicit summary  -> {summary_path}")
    print(f"Saved dark-market plot -> {fig_dir / 'elliptic_dark_market_f1.png'}")
    print("\nIllicit-class test summary:")
    print(summary.round(4).to_string(index=False))


def _plot_dark_market(per_step: pd.DataFrame, model_names, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for name in model_names:
        ax.plot(per_step["time_step"], per_step[name], marker="o", ms=3, label=name)
    ax.axvline(DARK_MARKET_STEP, color="red", ls="--", lw=1.2,
               label=f"dark-market shutdown (step {DARK_MARKET_STEP})")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Illicit-class F1")
    ax.set_title("Elliptic — illicit-detection F1 over time (dark-market shutdown)")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    run(load_config("configs/elliptic.yaml"))
