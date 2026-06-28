"""Loading and splitting for the German Credit Data (credit-risk track).

The dataset is a single small CSV (~1000 applicants) of credit-scoring
attributes with a good/bad risk label. Unlike the fraud tracks there is no
sub-sampling (the whole dataset fits in memory trivially) and no graph.

The Kaggle file circulates in two schemas:

- the *cleaned* version (``Age, Sex, Job, Housing, Saving accounts,
  Checking account, Credit amount, Duration, Purpose``) with a ``Risk``
  column valued ``good`` / ``bad``; and
- the original UCI Statlog version, whose target is coded ``1`` (good) /
  ``2`` (bad).

There is no applicant id, so we add a synthetic ``ApplicantID`` (the row
position) used to align feature rows with split membership. The target is
normalised to **bad = 1, good = 0** regardless of the source encoding, since
"bad" (default) is the event we want KS / Gini to separate. See
``data/README.md`` for the raw file layout.
"""

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.io import load_config
from src.utils.seed import set_seed

ID_COL = "ApplicantID"


def _normalize_label(series: pd.Series) -> pd.Series:
    """Map the risk target onto bad = 1 / good = 0, whatever its source encoding.

    Handles the cleaned (``good`` / ``bad`` strings) and UCI Statlog
    (``1`` good / ``2`` bad integer codes) versions, and is a no-op for data
    already coded ``0`` / ``1``.
    """
    if not pd.api.types.is_numeric_dtype(series):
        mapping = {"bad": 1, "good": 0}
        mapped = series.astype(str).str.strip().str.lower().map(mapping)
        if mapped.isna().any():
            unknown = sorted(series[mapped.isna()].astype(str).unique())
            raise ValueError(f"Unrecognised risk labels: {unknown}")
        return mapped.astype(int)

    uniques = set(series.dropna().unique())
    if uniques <= {1, 2}:  # UCI Statlog: 1 = good, 2 = bad
        return (series == 2).astype(int)
    if uniques <= {0, 1}:  # already in the target convention
        return series.astype(int)
    raise ValueError(f"Unrecognised numeric risk encoding: {sorted(uniques)}")


def _resolve_label_column(df: pd.DataFrame, label_col: str | None) -> str:
    """Return the label column name, accepting common German-credit aliases."""
    if label_col is not None and label_col in df.columns:
        return label_col

    candidates = []
    if label_col is not None:
        candidates.append(label_col)
    candidates.extend(
        [
            "Risk",
            "risk",
            "Creditability",
            "creditability",
            "target",
            "Target",
            "label",
            "Label",
            "class",
            "Class",
        ]
    )

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    raise ValueError(
        "No target/label column found in the German Credit CSV. Expected a column "
        "such as 'Risk' or 'Creditability'. The provided file appears to contain "
        "only feature columns, so the baseline pipeline cannot create splits or "
        "evaluate KS/Gini metrics."
    )


def load_raw(
    raw_dir: str | Path,
    data_file: str = "german_credit_data.csv",
    label_col: str | None = "Risk",
) -> pd.DataFrame:
    """Load the German Credit CSV, drop the export index, add a synthetic id,
    and normalise the target to bad = 1 / good = 0."""
    raw_dir = Path(raw_dir)
    df = pd.read_csv(raw_dir / data_file)

    # The cleaned Kaggle export ships a leading unnamed row-index column.
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")

    resolved_label_col = _resolve_label_column(df, label_col)
    target_col = label_col or "Risk"
    if resolved_label_col != target_col:
        df[target_col] = df[resolved_label_col]
        df = df.drop(columns=[resolved_label_col])

    df[target_col] = _normalize_label(df[target_col])

    df.insert(0, ID_COL, range(len(df)))
    return df


def make_splits(
    df: pd.DataFrame,
    label_col: str = "Risk",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Stratified train/val/test split on ``label_col`` (original index preserved)."""
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
    """Convenience wrapper: load raw data and split (no sub-sampling)."""
    set_seed(config["seed"])

    paths = config["paths"]
    df = load_raw(
        raw_dir=paths["raw_dir"],
        data_file=paths["data_file"],
        label_col=config["features"]["label_col"],
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


def save_interim(df: pd.DataFrame, interim_dir: str | Path, filename: str = "clean.parquet") -> Path:
    """Persist the label-normalised full dataframe to ``data/interim/german_credit/``.

    The trainer's cross-validation path reloads this (the whole dataset) so it
    can refit the feature pipeline inside each fold.
    """
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / filename
    df.to_parquet(out_path)
    return out_path


if __name__ == "__main__":
    cfg = load_config("configs/german_credit.yaml")
    df, splits = load_and_split(cfg)

    out_path = save_interim(df, cfg["paths"]["interim_dir"])
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols -> {out_path}")

    for name, split_df in splits.items():
        bad_rate = split_df[cfg["split"]["stratify_col"]].mean()
        print(f"{name}: {len(split_df)} rows, bad rate = {bad_rate:.4%}")
