"""Feature engineering for the German Credit Data (credit-risk track).

``GermanCreditFeaturePipeline`` mirrors ``PaySimFeaturePipeline`` /
``IEEECISFeaturePipeline``: it is fit once on the training split and reused
(via :meth:`save` / :meth:`load`) to transform val/test, so the baselines all
see identical feature definitions. For each numeric column it produces a *raw*
(sentinel-filled) view for tree models and a *scaled* (median-filled +
standardized) view for the linear model; categoricals (including the NaN-bearing
``Saving accounts`` / ``Checking account``) are label-encoded with an explicit
``"missing"`` category.

There are no high-cardinality identifier columns here, so (unlike PaySim) there
is no frequency encoding and no derived balance features.

On top of the model-ready matrix this module provides the credit-scoring
**Weight-of-Evidence (WoE)** and **Information Value (IV)** analysis that risk
teams use for feature screening:

- ``WoE = ln(%good / %bad)`` per bin (good = label 0, bad = label 1).
- ``IV  = sum_bins (%good - %bad) * WoE`` per feature.

Rule of thumb for IV: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium,
0.3-0.5 strong, >0.5 suspiciously strong (possible leakage).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class GermanCreditFeaturePipeline:
    def __init__(
        self,
        label_col: str,
        id_col: str,
        categorical_cols: List[str],
        log1p_cols: List[str],
        missing_value: float = -999.0,
    ) -> None:
        self.label_col = label_col
        self.id_col = id_col
        self.categorical_cols = categorical_cols
        self.log1p_cols = log1p_cols
        self.missing_value = missing_value

        self.numeric_cols_: List[str] = []
        self.label_encoders_: Dict[str, Dict[str, int]] = {}
        self.medians_: pd.Series | None = None
        self.scaler_: StandardScaler | None = None

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "GermanCreditFeaturePipeline":
        exclude = set(self.categorical_cols) | {self.label_col, self.id_col}
        self.numeric_cols_ = [c for c in df.columns if c not in exclude]

        for col in self.categorical_cols:
            self.label_encoders_[col] = self._fit_categorical(df[col])

        numeric_raw = self._apply_log1p(df[self.numeric_cols_].copy())
        self.medians_ = numeric_raw.median(numeric_only=True)
        numeric_for_scaling = numeric_raw.fillna(self.medians_)

        self.scaler_ = StandardScaler().fit(numeric_for_scaling.values)
        return self

    @staticmethod
    def _fit_categorical(series: pd.Series) -> Dict[str, int]:
        values = series.fillna("missing").astype(str)
        categories = sorted(values.unique())
        mapping = {cat: i for i, cat in enumerate(categories)}
        mapping["__unknown__"] = len(mapping)
        return mapping

    def _apply_log1p(self, numeric_df: pd.DataFrame) -> pd.DataFrame:
        for col in self.log1p_cols:
            if col in numeric_df.columns:
                numeric_df[col] = np.log1p(numeric_df[col].clip(lower=0))
        return numeric_df

    # ------------------------------------------------------------------
    # transforming
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform ``df`` into model-ready features (raw + scaled views)."""
        if self.scaler_ is None:
            raise RuntimeError("Pipeline must be fit before calling transform().")

        out = pd.DataFrame(index=df.index)
        out[self.id_col] = df[self.id_col].values
        if self.label_col in df.columns:
            out[self.label_col] = df[self.label_col].values

        for col in self.categorical_cols:
            out[col] = self._transform_categorical(df[col], self.label_encoders_[col])

        numeric_raw = self._apply_log1p(df[self.numeric_cols_].copy())

        numeric_tree = numeric_raw.fillna(self.missing_value)
        for col in self.numeric_cols_:
            out[col] = numeric_tree[col].values

        numeric_for_scaling = numeric_raw.fillna(self.medians_)
        scaled = self.scaler_.transform(numeric_for_scaling.values)
        for i, col in enumerate(self.numeric_cols_):
            out[f"{col}_scaled"] = scaled[:, i]

        return out

    @staticmethod
    def _transform_categorical(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
        values = series.fillna("missing").astype(str)
        unknown_idx = mapping["__unknown__"]
        return values.map(mapping).fillna(unknown_idx).astype(int)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # column groupings for downstream consumers
    # ------------------------------------------------------------------
    def tree_feature_cols(self) -> List[str]:
        """Feature columns for tree-based models (raw numeric + categorical codes)."""
        return self.categorical_cols + self.numeric_cols_

    def scaled_feature_cols(self) -> List[str]:
        """Feature columns for the linear model (scaled numeric + categorical codes)."""
        return self.categorical_cols + [f"{c}_scaled" for c in self.numeric_cols_]

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "GermanCreditFeaturePipeline":
        with open(path, "rb") as f:
            return pickle.load(f)


# ----------------------------------------------------------------------
# Weight-of-Evidence / Information Value (credit-scoring feature screening)
# ----------------------------------------------------------------------
def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    label_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-bin WoE table for one feature (good = label 0, bad = label 1).

    Numeric features are binned into up to ``n_bins`` equal-frequency bins;
    categorical / low-cardinality features use their raw values as bins. A
    small ``eps`` smooths bins that contain only goods or only bads so the
    log-ratio stays finite.
    """
    y = df[label_col].astype(int)
    x = df[feature]

    is_numeric = pd.api.types.is_numeric_dtype(x) and x.nunique(dropna=True) > n_bins
    if is_numeric:
        binned = pd.qcut(x, q=n_bins, duplicates="drop")
        bins = binned.cat.add_categories(["missing"]).fillna("missing")
    else:
        bins = x.fillna("missing").astype(str)

    total_good = int((y == 0).sum())
    total_bad = int((y == 1).sum())
    eps = 0.5  # Laplace-style smoothing for empty cells

    grouped = pd.DataFrame({"bin": bins, "y": y}).groupby("bin", observed=True)
    table = pd.DataFrame(
        {
            "n": grouped["y"].size(),
            "n_bad": grouped["y"].sum(),
        }
    ).reset_index()
    table.insert(0, "feature", feature)
    table["n_good"] = table["n"] - table["n_bad"]
    table["bad_rate"] = table["n_bad"] / table["n"]

    pct_good = (table["n_good"] + eps) / (total_good + eps)
    pct_bad = (table["n_bad"] + eps) / (total_bad + eps)
    table["woe"] = np.log(pct_good / pct_bad)
    table["iv_contrib"] = (pct_good - pct_bad) * table["woe"]
    return table


def information_value_summary(
    df: pd.DataFrame,
    features: List[str],
    label_col: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    """One row per feature: total IV and the standard predictive-power band.

    Sorted strongest-first so the dissertation can quote the top discriminators.
    """
    rows = []
    for feat in features:
        woe_table = compute_woe_iv(df, feat, label_col, n_bins=n_bins)
        iv = float(woe_table["iv_contrib"].sum())
        rows.append({"feature": feat, "iv": iv, "strength": _iv_band(iv)})
    return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)


def _iv_band(iv: float) -> str:
    if iv < 0.02:
        return "useless"
    if iv < 0.1:
        return "weak"
    if iv < 0.3:
        return "medium"
    if iv < 0.5:
        return "strong"
    return "suspicious"


if __name__ == "__main__":
    from src.data.german_credit import load_and_split
    from src.utils.io import load_config

    # Re-import under the dotted path so the pickled pipeline's class is
    # `src.features.german_credit_features....` rather than `__main__....`,
    # otherwise pickle.load() fails from other entry points.
    from src.features.german_credit_features import GermanCreditFeaturePipeline

    cfg = load_config("configs/german_credit.yaml")
    _, splits = load_and_split(cfg)

    feat_cfg = cfg["features"]
    pipeline = GermanCreditFeaturePipeline(
        label_col=feat_cfg["label_col"],
        id_col=feat_cfg["id_col"],
        categorical_cols=feat_cfg["categorical_cols"],
        log1p_cols=feat_cfg["log1p_cols"],
        missing_value=feat_cfg["missing_value"],
    )

    train_feats = pipeline.fit_transform(splits["train"])
    val_feats = pipeline.transform(splits["val"])
    test_feats = pipeline.transform(splits["test"])

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    for name, feats in [("train", train_feats), ("val", val_feats), ("test", test_feats)]:
        out_path = processed_dir / f"tabular_{name}.parquet"
        feats.to_parquet(out_path)
        print(f"{name}: {feats.shape} -> {out_path}")

    pipeline.save(processed_dir / "feature_pipeline.pkl")
    print(f"Saved fitted feature pipeline -> {processed_dir / 'feature_pipeline.pkl'}")

    # WoE / IV is computed on the TRAIN split only (screening must not peek at
    # val/test) and uses the original, human-readable column values.
    iv_summary = information_value_summary(
        splits["train"], feat_cfg["iv_cols"], feat_cfg["label_col"]
    )
    iv_path = Path(cfg["paths"]["iv_csv"])
    iv_path.parent.mkdir(parents=True, exist_ok=True)
    iv_summary.to_csv(iv_path, index=False)
    print(f"\nInformation Value (train split) -> {iv_path}")
    print(iv_summary.to_string(index=False))
