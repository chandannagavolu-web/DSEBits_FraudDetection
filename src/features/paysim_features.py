"""Feature engineering for the PaySim mobile-money dataset.

``PaySimFeaturePipeline`` mirrors ``IEEECISFeaturePipeline``: it is fit once
on the training split and reused (via :meth:`save` / :meth:`load`) to
transform val/test and to build per-node features for the heterogeneous
graph (``src/graph/paysim_graph.py``), so baselines and the GNN see identical
feature definitions.

On top of the raw PaySim columns it derives the balance-consistency features
that are the strongest fraud signal in this dataset:

- ``errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg`` — should be
  ~0 for a legitimate debit; fraudulent transfers frequently violate it.
- ``errorBalanceDest = oldbalanceDest + amount - newbalanceDest`` — the
  receiver-side analogue.
- ``balanceDeltaOrig`` / ``balanceDeltaDest`` — signed balance movements.
- ``hour = step % 24`` and ``day = step // 24`` — the simulation clock
  (``step`` is one hour) split into intra-day / day components.
- ``isMerchantDest`` — PaySim merchant accounts are prefixed ``"M"``.

For each numeric column the pipeline produces a *raw* (sentinel-filled) view
for tree models and a *scaled* (median-filled + standardized) view for
linear / GNN models. The categorical ``type`` is label-encoded; the
high-cardinality account ids are frequency-encoded (transaction degree).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add PaySim-specific derived columns. Pure: returns a new DataFrame."""
    out = df.copy()
    out["errorBalanceOrig"] = out["newbalanceOrig"] + out["amount"] - out["oldbalanceOrg"]
    out["errorBalanceDest"] = out["oldbalanceDest"] + out["amount"] - out["newbalanceDest"]
    out["balanceDeltaOrig"] = out["oldbalanceOrg"] - out["newbalanceOrig"]
    out["balanceDeltaDest"] = out["newbalanceDest"] - out["oldbalanceDest"]
    out["hour"] = out["step"] % 24
    out["day"] = out["step"] // 24
    out["isMerchantDest"] = out["nameDest"].astype(str).str.startswith("M").astype(int)
    return out


class PaySimFeaturePipeline:
    def __init__(
        self,
        label_col: str,
        id_col: str,
        categorical_cols: List[str],
        freq_encode_cols: List[str],
        log1p_cols: List[str],
        drop_cols: List[str] | None = None,
        missing_value: float = -999.0,
    ) -> None:
        self.label_col = label_col
        self.id_col = id_col
        self.categorical_cols = categorical_cols
        self.freq_encode_cols = freq_encode_cols
        self.log1p_cols = log1p_cols
        # Columns excluded from the numeric feature matrix (raw account id
        # strings — kept only for freq-encoding / graph construction — and the
        # leaky isFlaggedFraud business rule).
        self.drop_cols = drop_cols or []
        self.missing_value = missing_value

        self.numeric_cols_: List[str] = []
        self.label_encoders_: Dict[str, Dict[str, int]] = {}
        self.freq_maps_: Dict[str, Dict[str, int]] = {}
        self.medians_: pd.Series | None = None
        self.scaler_: StandardScaler | None = None

    # ------------------------------------------------------------------
    # fitting
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "PaySimFeaturePipeline":
        df = engineer_features(df)

        exclude = (
            set(self.categorical_cols)
            | set(self.freq_encode_cols)
            | set(self.drop_cols)
            | {self.label_col, self.id_col}
        )
        self.numeric_cols_ = [c for c in df.columns if c not in exclude]

        for col in self.categorical_cols:
            self.label_encoders_[col] = self._fit_categorical(df[col])

        for col in self.freq_encode_cols:
            self.freq_maps_[col] = self._fit_freq(df[col])

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

    @staticmethod
    def _fit_freq(series: pd.Series) -> Dict[str, int]:
        values = series.fillna("missing").astype(str)
        return values.value_counts().to_dict()

    def _apply_log1p(self, numeric_df: pd.DataFrame) -> pd.DataFrame:
        for col in self.log1p_cols:
            if col in numeric_df.columns:
                numeric_df[col] = np.log1p(numeric_df[col].clip(lower=0))
        return numeric_df

    # ------------------------------------------------------------------
    # transforming
    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame, include_raw: bool = True) -> pd.DataFrame:
        """Transform ``df`` into model-ready features.

        Args:
            include_raw: if ``False``, skip the sentinel-filled "raw"/tree
                view and store the scaled view as float32 — used by graph
                construction (which only consumes ``scaled_feature_cols()``)
                to roughly halve peak memory.
        """
        if self.scaler_ is None:
            raise RuntimeError("Pipeline must be fit before calling transform().")

        df = engineer_features(df)

        out = pd.DataFrame(index=df.index)
        out[self.id_col] = df[self.id_col].values
        if self.label_col in df.columns:
            out[self.label_col] = df[self.label_col].values

        for col in self.categorical_cols:
            out[col] = self._transform_categorical(df[col], self.label_encoders_[col])

        for col in self.freq_encode_cols:
            out[f"{col}_freq"] = self._transform_freq(df[col], self.freq_maps_[col])

        numeric_raw = self._apply_log1p(df[self.numeric_cols_].copy())

        if include_raw:
            numeric_tree = numeric_raw.fillna(self.missing_value)
            for col in self.numeric_cols_:
                out[col] = numeric_tree[col].values
            del numeric_tree

        numeric_for_scaling = numeric_raw.fillna(self.medians_)
        del numeric_raw
        scaled = self.scaler_.transform(numeric_for_scaling.values)
        del numeric_for_scaling
        if not include_raw:
            scaled = scaled.astype(np.float32)
        for i, col in enumerate(self.numeric_cols_):
            out[f"{col}_scaled"] = scaled[:, i]

        return out

    @staticmethod
    def _transform_categorical(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
        values = series.fillna("missing").astype(str)
        unknown_idx = mapping["__unknown__"]
        return values.map(mapping).fillna(unknown_idx).astype(int)

    @staticmethod
    def _transform_freq(series: pd.Series, freq_map: Dict[str, int]) -> pd.Series:
        values = series.fillna("missing").astype(str)
        return values.map(freq_map).fillna(0).astype(float)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # column groupings for downstream consumers
    # ------------------------------------------------------------------
    def tree_feature_cols(self) -> List[str]:
        """Feature columns for tree-based models (raw numeric + categorical codes + freq)."""
        return (
            self.categorical_cols
            + [f"{c}_freq" for c in self.freq_encode_cols]
            + self.numeric_cols_
        )

    def scaled_feature_cols(self) -> List[str]:
        """Feature columns for linear / GNN models (scaled numeric + categorical codes + freq)."""
        return (
            self.categorical_cols
            + [f"{c}_freq" for c in self.freq_encode_cols]
            + [f"{c}_scaled" for c in self.numeric_cols_]
        )

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "PaySimFeaturePipeline":
        with open(path, "rb") as f:
            return pickle.load(f)


if __name__ == "__main__":
    from src.data.paysim import load_and_split
    from src.utils.io import load_config

    # Re-import under the package's dotted path so the pickled pipeline's
    # class is `src.features.paysim_features.PaySimFeaturePipeline` rather
    # than `__main__....` — otherwise pickle.load() fails from other entry
    # points (e.g. train_baseline_paysim.py).
    from src.features.paysim_features import PaySimFeaturePipeline

    cfg = load_config("configs/paysim.yaml")
    _, splits = load_and_split(cfg)

    feat_cfg = cfg["features"]
    pipeline = PaySimFeaturePipeline(
        label_col=feat_cfg["label_col"],
        id_col=feat_cfg["id_col"],
        categorical_cols=feat_cfg["categorical_cols"],
        freq_encode_cols=feat_cfg["freq_encode_cols"],
        log1p_cols=feat_cfg["log1p_cols"],
        drop_cols=feat_cfg.get("drop_cols", []),
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
