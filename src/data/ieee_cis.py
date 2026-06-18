"""Loading, merging and splitting for the IEEE-CIS Fraud Detection dataset.

See ``data/README.md`` for the expected raw file layout and the rationale
for creating our own train/val/test split from ``train_transaction.csv`` /
``train_identity.csv`` (Kaggle's hidden test set has no public labels).
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import load_config
from src.utils.seed import set_seed


def load_raw(
    raw_dir: str | Path,
    transaction_file: str = "train_transaction.csv",
    identity_file: str = "train_identity.csv",
) -> pd.DataFrame:
    """Load and left-join the transaction and identity tables.

    Rows without a matching identity record get NaN in the identity
    columns (e.g. ``DeviceType``, ``DeviceInfo``, ``id_01``...``id_38``),
    which is handled downstream by feature engineering / graph construction.
    """
    raw_dir = Path(raw_dir)
    transactions = pd.read_csv(raw_dir / transaction_file)
    identity = pd.read_csv(raw_dir / identity_file)

    df = transactions.merge(identity, on="TransactionID", how="left")
    return df


def make_splits(
    df: pd.DataFrame,
    label_col: str = "isFraud",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Stratified train/val/test split on ``label_col``.

    Returns a dict with keys ``"train"``, ``"val"``, ``"test"``, each a
    DataFrame with the original index preserved (used later to align graph
    node masks with tabular feature rows).
    """
    if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    train_df, rest_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=df[label_col],
        random_state=seed,
    )

    val_ratio_within_rest = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        rest_df,
        train_size=val_ratio_within_rest,
        stratify=rest_df[label_col],
        random_state=seed,
    )

    return {"train": train_df, "val": val_df, "test": test_df}


def load_and_split(config: dict) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Convenience wrapper: load raw data and produce the configured splits."""
    set_seed(config["seed"])

    paths = config["paths"]
    df = load_raw(
        raw_dir=paths["raw_dir"],
        transaction_file=paths["transaction_file"],
        identity_file=paths["identity_file"],
    )

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


def save_interim(df: pd.DataFrame, interim_dir: str | Path, filename: str = "merged_clean.parquet") -> Path:
    """Persist the merged raw dataframe to ``data/interim/ieee_cis/``."""
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / filename
    df.to_parquet(out_path)
    return out_path


if __name__ == "__main__":
    cfg = load_config("configs/ieee_cis.yaml")
    merged = load_raw(
        raw_dir=cfg["paths"]["raw_dir"],
        transaction_file=cfg["paths"]["transaction_file"],
        identity_file=cfg["paths"]["identity_file"],
    )
    out_path = save_interim(merged, cfg["paths"]["interim_dir"])
    print(f"Merged {merged.shape[0]} rows x {merged.shape[1]} cols -> {out_path}")

    splits = make_splits(
        merged,
        label_col=cfg["split"]["stratify_col"],
        train_frac=cfg["split"]["train_frac"],
        val_frac=cfg["split"]["val_frac"],
        test_frac=cfg["split"]["test_frac"],
        seed=cfg["seed"],
    )
    for name, split_df in splits.items():
        fraud_rate = split_df[cfg["split"]["stratify_col"]].mean()
        print(f"{name}: {len(split_df)} rows, fraud rate = {fraud_rate:.4%}")
