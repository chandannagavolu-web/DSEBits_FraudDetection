"""Loading, cleaning and temporal splitting for the Lending Club Loan Data
(credit-risk track: tabular default prediction + temporal TCN).

Lending Club's accepted-loans file (~2.26M rows, ~151 columns) is the credit
track's large, real-world dataset. Two concerns dominate the loader:

1. **Leakage.** Most columns record post-origination outcomes (``recoveries``,
   ``total_pymnt*``, ``last_pymnt_*``, ``out_prncp*``, ``last_fico_*``, the
   hardship/settlement blocks) and would trivially reveal the label. We read
   only an **allowlist** of origination-time columns (``load.usecols``) plus the
   few fields needed to derive features or to split/label. A drop-list would be
   fragile on a 151-column file; an allowlist cannot leak by accident.

2. **Censoring.** ``loan_status`` includes in-progress loans (``Current``,
   ``Issued``, ``In Grace Period``, ``Late ...``) whose outcome is unknown. Only
   terminal statuses are kept: ``Fully Paid`` (good = 0) vs
   ``Charged Off`` / ``Default`` (bad = 1).

The split is **out-of-time** on ``issue_d`` (train on earlier vintages, test on
later ones), the realistic way a scorecard is validated — not the random
stratified split used by German Credit.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.io import load_config
from src.utils.seed import set_seed

ID_COL = "LoanID"


# ----------------------------------------------------------------------
# cleaning helpers (Lending Club ships several numeric fields as strings)
# ----------------------------------------------------------------------
def _parse_term(series: pd.Series) -> pd.Series:
    """'36 months' / ' 60 months' -> 36 / 60 (int months)."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+)", expand=False), errors="coerce"
    )


def _parse_emp_length(series: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, '3 years' -> 3, 'n/a' -> NaN."""
    text = series.astype(str).str.lower()
    out = text.str.extract(r"(\d+)", expand=False)
    out = pd.to_numeric(out, errors="coerce")
    out[text.str.contains("< 1")] = 0
    return out


def _parse_percent(series: pd.Series) -> pd.Series:
    """'13.56%' -> 13.56 ; already-numeric input passes through."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    return pd.to_numeric(
        series.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce"
    )


def _parse_month_year(series: pd.Series) -> pd.Series:
    """'Dec-2015' / 'Dec-15' -> datetime (first of month)."""
    return pd.to_datetime(series, format="%b-%Y", errors="coerce").fillna(
        pd.to_datetime(series, format="%b-%y", errors="coerce")
    )


def _clean_chunk(df: pd.DataFrame, target_cfg: dict) -> pd.DataFrame:
    """Filter to terminal-status loans, map the target, and engineer features."""
    bad = set(target_cfg["bad_statuses"])
    good = set(target_cfg["good_statuses"])

    status = df["loan_status"].astype(str)
    keep = status.isin(bad | good)
    df = df.loc[keep].copy()
    if df.empty:
        return df

    df["target"] = status[keep].isin(bad).astype(int)

    # Temporal split key + credit-age engineering.
    df["issue_date"] = _parse_month_year(df["issue_d"])
    earliest = _parse_month_year(df["earliest_cr_line"])
    df["credit_age_mths"] = (
        (df["issue_date"].dt.year - earliest.dt.year) * 12
        + (df["issue_date"].dt.month - earliest.dt.month)
    )

    # Numeric parsing of the string-coded fields.
    df["term_months"] = _parse_term(df["term"])
    df["emp_length_years"] = _parse_emp_length(df["emp_length"])
    df["int_rate"] = _parse_percent(df["int_rate"])
    df["revol_util"] = _parse_percent(df["revol_util"])

    # FICO: the file gives a low/high band at origination; use the midpoint.
    df["fico"] = (
        pd.to_numeric(df["fico_range_low"], errors="coerce")
        + pd.to_numeric(df["fico_range_high"], errors="coerce")
    ) / 2.0

    return df


def load_raw(config: dict) -> pd.DataFrame:
    """Read the allowlisted columns in chunks, clean, and concatenate.

    Returns a dataframe of origination-time features + ``target`` +
    ``issue_date`` + a synthetic ``LoanID`` (row position) for split alignment.
    """
    paths = config["paths"]
    load_cfg = config["load"]
    raw_path = Path(paths["raw_dir"]) / paths["data_file"]

    # Intersect the allowlist with what the file actually has (schema varies
    # slightly across Lending Club releases).
    header = pd.read_csv(raw_path, nrows=0)
    usecols = [c for c in load_cfg["usecols"] if c in header.columns]

    cleaned: List[pd.DataFrame] = []
    reader = pd.read_csv(
        raw_path,
        usecols=usecols,
        chunksize=load_cfg.get("chunksize", 200000),
        low_memory=False,
    )
    for chunk in reader:
        cleaned.append(_clean_chunk(chunk, config["target"]))

    df = pd.concat(cleaned, ignore_index=True)
    df.insert(0, ID_COL, range(len(df)))
    return df


def make_temporal_splits(
    df: pd.DataFrame,
    date_col: str = "issue_date",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    """Out-of-time split: order by ``date_col`` and slice by cumulative fraction.

    The earliest ``train_frac`` of loans (by issue date) become train, the next
    ``val_frac`` val, the latest ``test_frac`` test — so the model is always
    validated on vintages it could not have seen at training time. Ties on the
    boundary date are broken by ``LoanID`` to keep the split deterministic.
    """
    if not abs(train_frac + val_frac + test_frac - 1.0) < 1e-6:
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    ordered = df.sort_values([date_col, ID_COL], kind="mergesort")
    n = len(ordered)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    return {
        "train": ordered.iloc[:n_train],
        "val": ordered.iloc[n_train : n_train + n_val],
        "test": ordered.iloc[n_train + n_val :],
    }


def load_and_split(config: dict) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Load + clean + out-of-time split, driven by the config."""
    set_seed(config["seed"])
    df = load_raw(config)

    split_cfg = config["split"]
    splits = make_temporal_splits(
        df,
        date_col=split_cfg["date_col"],
        train_frac=split_cfg["train_frac"],
        val_frac=split_cfg["val_frac"],
        test_frac=split_cfg["test_frac"],
    )
    return df, splits


def save_interim(df: pd.DataFrame, interim_dir: str | Path, filename: str = "clean.parquet") -> Path:
    interim_dir = Path(interim_dir)
    interim_dir.mkdir(parents=True, exist_ok=True)
    out_path = interim_dir / filename
    df.to_parquet(out_path)
    return out_path


if __name__ == "__main__":
    cfg = load_config("configs/lending_club.yaml")
    df, splits = load_and_split(cfg)

    out_path = save_interim(df, cfg["paths"]["interim_dir"])
    print(f"Loaded {df.shape[0]} terminal-status loans x {df.shape[1]} cols -> {out_path}")
    print(f"  date range: {df['issue_date'].min()} .. {df['issue_date'].max()}")
    for name, split_df in splits.items():
        bad_rate = split_df[cfg["split"]["stratify_col"]].mean()
        lo, hi = split_df["issue_date"].min(), split_df["issue_date"].max()
        print(f"  {name}: {len(split_df)} loans, bad rate = {bad_rate:.4%}, vintages {lo:%Y-%m}..{hi:%Y-%m}")
