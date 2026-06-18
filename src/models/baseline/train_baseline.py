"""Train and evaluate traditional ML baselines on the IEEE-CIS tabular features.

Baselines: Logistic Regression, Random Forest, XGBoost — benchmarked against
the GNN model (``src/models/gnn/train_gnn.py``) using the shared metrics in
``src/evaluation/metrics.py``.

Usage:
    python -m src.models.baseline.train_baseline
"""

from pathlib import Path
from typing import List, Tuple

import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import compute_all_metrics
from src.features.ieee_cis_features import IEEECISFeaturePipeline
from src.utils.io import load_config, log_results
from src.utils.seed import set_seed


def load_processed(processed_dir: str | Path, use_smote: bool = False):
    processed_dir = Path(processed_dir)

    train_file = "tabular_train_smote.parquet" if use_smote else "tabular_train.parquet"
    train = pd.read_parquet(processed_dir / train_file)
    val = pd.read_parquet(processed_dir / "tabular_val.parquet")
    test = pd.read_parquet(processed_dir / "tabular_test.parquet")
    pipeline = IEEECISFeaturePipeline.load(processed_dir / "feature_pipeline.pkl")
    return train, val, test, pipeline


def feature_cols_for(model_name: str, pipeline: IEEECISFeaturePipeline) -> List[str]:
    """Logistic regression uses scaled features; tree models use raw features."""
    if model_name == "logistic_regression":
        return pipeline.scaled_feature_cols()
    return pipeline.tree_feature_cols()


def build_model(model_name: str, params: dict, train: pd.DataFrame, label_col: str, use_smote: bool = False):
    params = dict(params)
    # SMOTE already balances the training set, so class-weighting /
    # scale_pos_weight on top of it would double-correct for imbalance.
    if model_name == "logistic_regression":
        if use_smote:
            params.pop("class_weight", None)
        return LogisticRegression(**params)
    if model_name == "random_forest":
        if use_smote:
            params.pop("class_weight", None)
        return RandomForestClassifier(**params)
    if model_name == "xgboost":
        if not use_smote:
            pos = train[label_col].sum()
            neg = len(train) - pos
            params["scale_pos_weight"] = neg / max(pos, 1)
        return xgb.XGBClassifier(**params)
    raise ValueError(f"Unknown baseline model: {model_name}")


def evaluate(model, df: pd.DataFrame, cols: List[str], label_col: str) -> dict:
    y_true = df[label_col].values
    y_score = model.predict_proba(df[cols])[:, 1]
    return compute_all_metrics(y_true, y_score)


def run(cfg: dict) -> None:
    set_seed(cfg["seed"])

    use_smote = cfg.get("resampling", {}).get("enabled", False)
    train, val, test, pipeline = load_processed(cfg["paths"]["processed_dir"], use_smote=use_smote)
    label_col = cfg["features"]["label_col"]

    model_dir = Path(cfg["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    results_csv = cfg["paths"]["results_csv"]

    for model_name in cfg["baseline"]["models"]:
        params = cfg["baseline"][model_name]
        cols = feature_cols_for(model_name, pipeline)

        model = build_model(model_name, params, train, label_col, use_smote=use_smote)
        model.fit(train[cols], train[label_col])

        for split_name, split_df in [("val", val), ("test", test)]:
            metrics = evaluate(model, split_df, cols, label_col)
            log_results(
                {
                    "experiment": cfg["experiment_name"],
                    "model": model_name,
                    "resampling": "smote" if use_smote else "none",
                    "split": split_name,
                    "n_features": len(cols),
                    **metrics,
                },
                results_csv,
            )
            print(f"[{model_name}] {split_name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        out_path = model_dir / f"{model_name}.json" if model_name == "xgboost" else model_dir / f"{model_name}.pkl"
        if model_name == "xgboost":
            model.save_model(out_path)
        else:
            import pickle

            with open(out_path, "wb") as f:
                pickle.dump(model, f)
        print(f"Saved {model_name} -> {out_path}")


if __name__ == "__main__":
    config = load_config("configs/ieee_cis.yaml")
    run(config)
