"""Credit-scoring KS / Gini scorecard for the trained Lending Club baselines.

The headline evaluation for the Lending Club tabular track. It reloads every
saved baseline (Logistic Regression, Random Forest, XGBoost), regenerates
out-of-time **test**-vintage scores, and produces the credit-scorecard view from
:mod:`src.evaluation.ks_gini`:

- a per-model decile KS table  -> reports/ks_gini/<model>_lending_club_ks_table.csv
- a combined KS / Gini summary -> reports/ks_gini/lending_club_ks_gini_summary.csv
- a per-model KS curve plot     -> reports/figures/ks_<model>_lending_club.png

Unlike German Credit (a ~150-row test split scored over 5 bands), Lending Club's
test set is large, so the full 10 deciles are used.

Run after training:
    python -m src.evaluation.evaluate_lending_club_ks_gini
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import pandas as pd
import xgboost as xgb

from src.evaluation.ks_gini import ks_decile_table, ks_gini_summary, plot_ks_curve
from src.models.baseline.train_baseline_lending_club import feature_cols_for, load_processed
from src.utils.io import load_config
from src.utils.seed import set_seed


def _load_model(model_name: str, model_dir: Path):
    if model_name == "xgboost":
        clf = xgb.XGBClassifier()
        clf.load_model(model_dir / f"{model_name}.json")
        return clf
    with open(model_dir / f"{model_name}.pkl", "rb") as f:
        return pickle.load(f)


def score_models(cfg: dict, test: pd.DataFrame, pipeline) -> Dict[str, dict]:
    label_col = cfg["features"]["label_col"]
    model_dir = Path(cfg["paths"]["model_dir"])

    out: Dict[str, dict] = {}
    for model_name in cfg["baseline"]["models"]:
        cols = feature_cols_for(model_name, pipeline)
        model = _load_model(model_name, model_dir)
        y_score = model.predict_proba(test[cols])[:, 1]
        out[model_name] = {"y_true": test[label_col].to_numpy(), "y_score": y_score}
    return out


def run(cfg: dict) -> None:
    set_seed(cfg["seed"])

    _, _, test, pipeline = load_processed(cfg["paths"]["processed_dir"])
    scored = score_models(cfg, test, pipeline)

    out_dir = Path(cfg["paths"]["results_csv"]).parent / "ks_gini"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("reports/figures")

    summary_rows: List[dict] = []
    for model_name, d in scored.items():
        y_true, y_score = d["y_true"], d["y_score"]

        table = ks_decile_table(y_true, y_score, n_bands=10)
        table_path = out_dir / f"{model_name}_lending_club_ks_table.csv"
        table.to_csv(table_path, index=False)

        plot_ks_curve(
            table,
            title=f"Lending Club KS curve — {model_name}",
            out_path=fig_dir / f"ks_{model_name}_lending_club.png",
        )

        s = ks_gini_summary(y_true, y_score, n_bands=10)
        summary_rows.append({"model": model_name, "split": "test", **s})
        print(
            f"[{model_name}] Gini={s['gini']:.4f}  KS_decile={s['ks_decile']:.4f}  "
            f"KS_cont={s['ks_continuous']:.4f}  peak_band={s['ks_peak_band']}  -> {table_path.name}"
        )

    summary = pd.DataFrame(summary_rows).sort_values("gini", ascending=False)
    summary_path = out_dir / "lending_club_ks_gini_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nSaved KS/Gini summary -> {summary_path}")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    run(load_config("configs/lending_club.yaml"))
