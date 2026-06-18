"""SMOTE-based class-imbalance correction for the IEEE-CIS training split.

Run *after* ``src/features/ieee_cis_features.py`` so it operates on the
fully-numeric feature table the baselines already use. Only the training
split is resampled — validation and test keep the original ~3.5% fraud
rate so PR-AUC / MCC / KS / Gini reflect real-world performance.

Usage:
    python -m src.features.resampling
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
from imblearn.over_sampling import SMOTE

from src.features.ieee_cis_features import IEEECISFeaturePipeline
from src.utils.io import load_config
from src.utils.seed import set_seed


def apply_smote(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    sampling_strategy: float | str = "auto",
    k_neighbors: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Return ``feature_cols`` + ``label_col`` with synthetic minority rows added.

    Synthetic rows are interpolated between minority-class neighbours, so
    the id column and any columns outside ``feature_cols`` are dropped —
    they have no meaningful value for a synthetic transaction.
    """
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        k_neighbors=k_neighbors,
        random_state=random_state,
    )
    # float32 halves peak memory during synthetic-sample generation, which
    # matters at this row/feature count.
    X = df[feature_cols].astype("float32")
    X_res, y_res = smote.fit_resample(X, df[label_col])

    out = X_res.copy()
    out[label_col] = y_res.values
    return out.reset_index(drop=True)


if __name__ == "__main__":
    cfg = load_config("configs/ieee_cis.yaml")
    set_seed(cfg["seed"])

    processed_dir = Path(cfg["paths"]["processed_dir"])
    train = pd.read_parquet(processed_dir / "tabular_train.parquet")
    pipeline = IEEECISFeaturePipeline.load(processed_dir / "feature_pipeline.pkl")

    label_col = cfg["features"]["label_col"]
    resample_cfg = cfg["resampling"]

    # Union of the tree (raw) and linear/GNN (scaled) feature views, so a
    # single resampled table serves both baseline families.
    feature_cols = sorted(set(pipeline.tree_feature_cols()) | set(pipeline.scaled_feature_cols()))

    before = train[label_col].value_counts().to_dict()
    resampled = apply_smote(
        train,
        feature_cols=feature_cols,
        label_col=label_col,
        sampling_strategy=resample_cfg["sampling_strategy"],
        k_neighbors=resample_cfg["k_neighbors"],
        random_state=cfg["seed"],
    )
    after = resampled[label_col].value_counts().to_dict()

    out_path = processed_dir / "tabular_train_smote.parquet"
    resampled.to_parquet(out_path)
    print(f"Class counts before SMOTE: {before}")
    print(f"Class counts after SMOTE:  {after}")
    print(f"Saved resampled training set -> {out_path}")
