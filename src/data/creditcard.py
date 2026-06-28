"""Loading and splitting for the Credit Card Fraud Detection dataset (ULB/Kaggle).

The autoencoder anomaly-detection track (CLAUDE.md Objective 3). The data is a
single CSV (~284,807 rows) of 30 numeric columns — ``Time``, the PCA-anonymised
``V1``..``V28``, and ``Amount`` — plus a binary ``Class`` (fraud = 1, ~0.172%).

There is no id column, so a synthetic ``TxnID`` (row position) is added to align
feature rows with split membership. The split is a standard stratified
train/val/test (this is a tabular anomaly-detection problem, not temporal/graph),
and a convenience helper returns the **normal-only** training rows the
autoencoder is fit on.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import load_config
from src.utils.seed import set_seed

ID_COL = "TxnID"


def load_raw(raw_dir: str | Path, data_file: str = "creditcard.csv", label_col: str = "Class") -> pd.DataFrame:
    """Load the credit-card CSV and attach a synthetic ``TxnID``."""
    raw_dir = Path(raw_dir)
    df = pd.read_csv(raw_dir / data_file)
    if label_col not in df.columns:
        raise ValueError(
            f"Expected a '{label_col}' column in {data_file}; found {list(df.columns)[:8]}..."
        )
    df[label_col] = df[label_col].astype(int)
    df.insert(0, ID_COL, range(len(df)))
    return df


def make_splits(
    df: pd.DataFrame,
    label_col: str = "Class",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Stratified train/val/test split on ``label_col`` (original index preserved)."""
    if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    train_df, rest_df = train_test_split(
        df, train_size=train_frac, stratify=df[label_col], random_state=seed
    )
    val_ratio = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        rest_df, train_size=val_ratio, stratify=rest_df[label_col], random_state=seed
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def normal_only(df: pd.DataFrame, label_col: str = "Class") -> pd.DataFrame:
    """Rows of the non-fraud (normal) class — what the autoencoder is fit on."""
    return df[df[label_col] == 0]


def load_and_split(config: dict) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    set_seed(config["seed"])
    paths = config["paths"]
    df = load_raw(paths["raw_dir"], paths["data_file"], config["features"]["label_col"])

    split_cfg = config["split"]
    splits = make_splits(
        df,
        label_col=split_cfg["stratify_col"],
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
        seed=config["seed"],
    )
    return df, splits


def save_interim(df: pd.DataFrame, interim_dir: str | Path, filename: str = "clean.parquet") -> Path:
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / filename
    df.to_parquet(out_path)
    return out_path


if __name__ == "__main__":
    cfg = load_config("configs/creditcard.yaml")
    df, splits = load_and_split(cfg)
    save_interim(df, cfg["paths"]["interim_dir"])
    print(f"Loaded {len(df)} transactions, fraud rate = {df['Class'].mean():.4%}")
    for name, sdf in splits.items():
        print(f"  {name}: {len(sdf)} rows, fraud rate = {sdf['Class'].mean():.4%}, "
              f"normal-only = {len(normal_only(sdf))}")
