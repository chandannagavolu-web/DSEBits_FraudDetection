"""Loading, (optional) sub-sampling and splitting for the PaySim dataset.

PaySim is distributed as a single CSV (``PS_2017..._log.csv``) of ~6.3M
simulated mobile-money transactions with columns::

    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
    nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Unlike IEEE-CIS there is no transaction id, so we add a synthetic
``TransactionID`` (the row position) used to align graph node masks with
tabular feature rows. See ``data/README.md`` for the raw file layout.

The full transaction graph does not fit in 8GB RAM, so an optional
*stratified* sub-sample (``subsample`` in ``configs/paysim.yaml``) is taken
BEFORE splitting — this keeps baselines and the GNN on identical rows for a
fair comparison and preserves the (rare) fraud rate.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import load_config
from src.utils.seed import set_seed

ID_COL = "TransactionID"


def load_raw(
    raw_dir: str | Path,
    transaction_file: str = "PS_20174392719_1491204439457_log.csv",
) -> pd.DataFrame:
    """Load the PaySim CSV and attach a synthetic ``TransactionID``.

    The id is the (stable) row position, giving every transaction a unique
    handle for split masks and graph node alignment.
    """
    raw_dir = Path(raw_dir)
    df = pd.read_csv(raw_dir / transaction_file)
    df.insert(0, ID_COL, range(len(df)))
    return df


def subsample(
    df: pd.DataFrame,
    fraction: float,
    label_col: str = "isFraud",
    seed: int = 42,
) -> pd.DataFrame:
    """Return a stratified ``fraction`` of ``df``, preserving the fraud rate.

    A no-op (returns ``df`` unchanged) when ``fraction >= 1.0``.
    """
    if fraction >= 1.0:
        return df
    sampled, _ = train_test_split(
        df,
        train_size=fraction,
        stratify=df[label_col],
        random_state=seed,
    )
    return sampled


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
    """Convenience wrapper: load raw data, sub-sample, and split."""
    set_seed(config["seed"])

    paths = config["paths"]
    df = load_raw(
        raw_dir=paths["raw_dir"],
        transaction_file=paths["transaction_file"],
    )

    split_cfg = config["split"]
    sub_cfg = config.get("subsample", {})
    if sub_cfg.get("enabled", False):
        df = subsample(
            df,
            fraction=sub_cfg["fraction"],
            label_col=split_cfg["stratify_col"],
            seed=config["seed"],
        )

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
    """Persist the (sub-sampled) raw dataframe to ``data/interim/paysim/``."""
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / filename
    df.to_parquet(out_path)
    return out_path


if __name__ == "__main__":
    cfg = load_config("configs/paysim.yaml")
    df, splits = load_and_split(cfg)

    out_path = save_interim(df, cfg["paths"]["interim_dir"])
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols -> {out_path}")

    for name, split_df in splits.items():
        fraud_rate = split_df[cfg["split"]["stratify_col"]].mean()
        print(f"{name}: {len(split_df)} rows, fraud rate = {fraud_rate:.4%}")
