"""Credit-scoring KS / Gini evaluation for all trained PaySim models.

Reloads every saved PaySim model (the full- and no-leakage tabular baselines
plus the heterogeneous GNN), regenerates test-set scores, and produces the
credit-scorecard view from :mod:`src.evaluation.ks_gini`:

- a per-model decile KS table  -> reports/ks_gini/<model>_<fset>_ks_table.csv
- a combined KS / Gini summary -> reports/ks_gini/ks_gini_summary.csv
- a per-model KS curve plot     -> reports/figures/ks_<model>_<fset>.png

Run after training:
    python -m src.evaluation.evaluate_paysim_ks_gini
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import xgboost as xgb

from src.evaluation.ks_gini import (
    ks_decile_table,
    ks_gini_summary,
    plot_ks_curve,
)
from src.models.baseline.train_baseline_paysim import feature_cols_for, load_processed
from src.utils.io import load_config
from src.utils.seed import set_seed

# (model_name, feature_set, exclude_leakage)
TABULAR_RUNS: List[Tuple[str, str, bool]] = [
    ("logistic_regression", "full", False),
    ("random_forest", "full", False),
    ("xgboost", "full", False),
    ("logistic_regression", "no_leakage", True),
    ("random_forest", "no_leakage", True),
    ("xgboost", "no_leakage", True),
]


def _load_tabular_model(model_name: str, feature_set: str, model_dir: Path):
    suffix = "_no_leakage" if feature_set == "no_leakage" else ""
    if model_name == "xgboost":
        clf = xgb.XGBClassifier()
        clf.load_model(model_dir / f"{model_name}{suffix}.json")
        return clf
    with open(model_dir / f"{model_name}{suffix}.pkl", "rb") as f:
        return pickle.load(f)


def score_tabular(cfg: dict, test: pd.DataFrame, pipeline) -> Dict[str, dict]:
    """Return {run_key: {y_true, y_score, feature_set, model}} for tabular models."""
    label_col = cfg["features"]["label_col"]
    leakage_cols = cfg["features"].get("leakage_cols", [])
    model_dir = Path(cfg["paths"]["model_dir"])

    out: Dict[str, dict] = {}
    for model_name, feature_set, exclude_leakage in TABULAR_RUNS:
        exclude = leakage_cols if exclude_leakage else None
        cols = feature_cols_for(model_name, pipeline, exclude=exclude)
        model = _load_tabular_model(model_name, feature_set, model_dir)
        y_score = model.predict_proba(test[cols])[:, 1]
        out[f"{model_name}__{feature_set}"] = {
            "model": model_name,
            "feature_set": feature_set,
            "y_true": test[label_col].to_numpy(),
            "y_score": y_score,
        }
    return out


def score_gnn(cfg: dict) -> Dict[str, dict]:
    """Reload the saved GNN state dict, rebuild the graph, score test nodes."""
    import torch

    from src.models.gnn.hetero_gnn import HeteroGNN

    torch.set_num_threads(2)
    gnn_cfg = cfg["gnn"]
    model_type = gnn_cfg["model_type"]
    model_path = Path(cfg["paths"]["model_dir"]) / f"{model_type}.pt"
    if not model_path.exists():
        print(f"[skip] GNN checkpoint not found: {model_path}")
        return {}

    graph = torch.load(Path(cfg["paths"]["processed_dir"]) / "graph.pt", weights_only=False)
    in_channels_dict = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    model = HeteroGNN(
        metadata=graph.metadata(),
        in_channels_dict=in_channels_dict,
        hidden_dim=gnn_cfg["hidden_dim"],
        num_layers=gnn_cfg["num_layers"],
        dropout=gnn_cfg["dropout"],
        model_type=model_type,
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(graph.x_dict, graph.edge_index_dict))

    test_mask = graph["transaction"].test_mask
    y_true = graph["transaction"].y[test_mask].cpu().numpy()
    y_score = probs[test_mask].cpu().numpy()
    # The graph carries the full feature pipeline (incl. engineered balance
    # features), so the GNN is comparable to the "full" tabular baselines.
    return {
        f"{model_type}__full": {
            "model": model_type,
            "feature_set": "full",
            "y_true": y_true,
            "y_score": y_score,
        }
    }


def run(cfg: dict) -> None:
    set_seed(cfg["seed"])

    _, _, test, pipeline = load_processed(cfg["paths"]["processed_dir"], use_smote=False)

    scored = score_tabular(cfg, test, pipeline)
    scored.update(score_gnn(cfg))

    out_dir = Path(cfg["paths"]["results_csv"]).parent / "ks_gini"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("reports/figures")

    summary_rows = []
    for key, d in scored.items():
        y_true, y_score = d["y_true"], d["y_score"]
        model, feature_set = d["model"], d["feature_set"]

        table = ks_decile_table(y_true, y_score, n_bands=10)
        table_path = out_dir / f"{model}_{feature_set}_ks_table.csv"
        table.to_csv(table_path, index=False)

        plot_ks_curve(
            table,
            title=f"PaySim KS curve — {model} ({feature_set})",
            out_path=fig_dir / f"ks_{model}_{feature_set}.png",
        )

        s = ks_gini_summary(y_true, y_score, n_bands=10)
        summary_rows.append(
            {"model": model, "feature_set": feature_set, "split": "test", **s}
        )
        print(
            f"[{model}/{feature_set}] "
            f"Gini={s['gini']:.4f}  KS_decile={s['ks_decile']:.4f}  "
            f"KS_cont={s['ks_continuous']:.4f}  peak_band={s['ks_peak_band']}  "
            f"-> {table_path.name}"
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        ["feature_set", "gini"], ascending=[True, False]
    )
    summary_path = out_dir / "ks_gini_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved KS/Gini summary -> {summary_path}")
    print(f"Saved decile tables + KS curves -> {out_dir} , {fig_dir}")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    run(load_config("configs/paysim.yaml"))
