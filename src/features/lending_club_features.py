"""Feature engineering for the Lending Club Loan Data (credit-risk track).

``LendingClubFeaturePipeline`` follows the same fit-on-train / reuse-on-val-test
contract as ``GermanCreditFeaturePipeline`` and ``PaySimFeaturePipeline``: each
numeric column gets a *raw* (sentinel-filled) view for tree models and a
*scaled* (median-filled + standardized) view for the linear model; low-card
categoricals are label-encoded with an explicit ``"missing"`` category. Unlike
German Credit, Lending Club has genuinely high-cardinality columns
(``addr_state``), so — as in the PaySim / IEEE-CIS pipelines — those are
**frequency-encoded** rather than given one integer code per value.

The Weight-of-Evidence / Information-Value screening helpers are shared with the
German Credit pipeline (imported, not duplicated), since they are dataset-
agnostic (they only need a feature column and a 0/1 label).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Re-exported so callers (and the dissertation's feature-analysis step) can pull
# the WoE/IV screening from one obvious place regardless of track.
from src.features.german_credit_features import (  # noqa: F401
    compute_woe_iv,
    information_value_summary,
)


class LendingClubFeaturePipeline:
    def __init__(
        self,
        label_col: str,
        id_col: str,
        numeric_cols: List[str],
        categorical_cols: List[str],
        freq_encode_cols: List[str],
        log1p_cols: List[str],
        missing_value: float = -999.0,
    ) -> None:
        self.label_col = label_col
        self.id_col = id_col
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.freq_encode_cols = list(freq_encode_cols)
        self.log1p_cols = list(log1p_cols)
        self.missing_value = missing_value

        self.label_encoders_: Dict[str, Dict[str, int]] = {}
        self.freq_maps_: Dict[str, Dict[str, float]] = {}
        self.medians_: pd.Series | None = None
        self.scaler_: StandardScaler | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, df: pd.DataFrame) -> "LendingClubFeaturePipeline":
        for col in self.categorical_cols:
            self.label_encoders_[col] = self._fit_categorical(df[col])
        for col in self.freq_encode_cols:
            counts = df[col].fillna("missing").astype(str).value_counts(normalize=True)
            self.freq_maps_[col] = counts.to_dict()

        numeric_raw = self._apply_log1p(df[self.numeric_cols].apply(pd.to_numeric, errors="coerce"))
        self.medians_ = numeric_raw.median(numeric_only=True)
        numeric_for_scaling = numeric_raw.fillna(self.medians_)
        self.scaler_ = StandardScaler().fit(numeric_for_scaling.values)
        return self

    @staticmethod
    def _fit_categorical(series: pd.Series) -> Dict[str, int]:
        values = series.fillna("missing").astype(str)
        mapping = {cat: i for i, cat in enumerate(sorted(values.unique()))}
        mapping["__unknown__"] = len(mapping)
        return mapping

    def _apply_log1p(self, numeric_df: pd.DataFrame) -> pd.DataFrame:
        for col in self.log1p_cols:
            if col in numeric_df.columns:
                numeric_df[col] = np.log1p(numeric_df[col].clip(lower=0))
        return numeric_df

    # ------------------------------------------------------------- transform
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scaler_ is None:
            raise RuntimeError("Pipeline must be fit before calling transform().")

        out = pd.DataFrame(index=df.index)
        out[self.id_col] = df[self.id_col].values
        if self.label_col in df.columns:
            out[self.label_col] = df[self.label_col].values

        for col in self.categorical_cols:
            out[col] = self._transform_categorical(df[col], self.label_encoders_[col])
        for col in self.freq_encode_cols:
            values = df[col].fillna("missing").astype(str)
            out[f"{col}_freq"] = values.map(self.freq_maps_[col]).fillna(0.0).astype(float)

        numeric_raw = self._apply_log1p(df[self.numeric_cols].apply(pd.to_numeric, errors="coerce"))

        numeric_tree = numeric_raw.fillna(self.missing_value)
        for col in self.numeric_cols:
            out[col] = numeric_tree[col].values

        scaled = self.scaler_.transform(numeric_raw.fillna(self.medians_).values)
        for i, col in enumerate(self.numeric_cols):
            out[f"{col}_scaled"] = scaled[:, i]

        return out

    @staticmethod
    def _transform_categorical(series: pd.Series, mapping: Dict[str, int]) -> pd.Series:
        values = series.fillna("missing").astype(str)
        return values.map(mapping).fillna(mapping["__unknown__"]).astype(int)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ----------------------------------------------- column groupings / io
    def _freq_cols(self) -> List[str]:
        return [f"{c}_freq" for c in self.freq_encode_cols]

    def tree_feature_cols(self) -> List[str]:
        """Raw numeric + categorical codes + frequency encodings (tree models)."""
        return self.categorical_cols + self._freq_cols() + self.numeric_cols

    def scaled_feature_cols(self) -> List[str]:
        """Scaled numeric + categorical codes + frequency encodings (linear/TCN)."""
        return self.categorical_cols + self._freq_cols() + [f"{c}_scaled" for c in self.numeric_cols]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "LendingClubFeaturePipeline":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_pipeline(cfg: dict) -> LendingClubFeaturePipeline:
    feat = cfg["features"]
    return LendingClubFeaturePipeline(
        label_col=feat["label_col"],
        id_col=feat["id_col"],
        numeric_cols=feat["numeric_cols"],
        categorical_cols=feat["categorical_cols"],
        freq_encode_cols=feat["freq_encode_cols"],
        log1p_cols=feat["log1p_cols"],
        missing_value=feat["missing_value"],
    )


if __name__ == "__main__":
    from src.data.lending_club import load_and_split, save_interim
    from src.utils.io import load_config

    # Re-import under the dotted path so the pickled pipeline's class is
    # `src.features.lending_club_features....` rather than `__main__....`.
    from src.features.lending_club_features import LendingClubFeaturePipeline  # noqa: F401

    cfg = load_config("configs/lending_club.yaml")
    df, splits = load_and_split(cfg)
    save_interim(df, cfg["paths"]["interim_dir"])

    pipeline = build_pipeline(cfg)
    train_feats = pipeline.fit_transform(splits["train"])
    val_feats = pipeline.transform(splits["val"])
    test_feats = pipeline.transform(splits["test"])

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, feats in [("train", train_feats), ("val", val_feats), ("test", test_feats)]:
        feats.to_parquet(processed_dir / f"tabular_{name}.parquet")
        print(f"{name}: {feats.shape} -> {processed_dir / f'tabular_{name}.parquet'}")
    pipeline.save(processed_dir / "feature_pipeline.pkl")

    iv_summary = information_value_summary(splits["train"], cfg["features"]["iv_cols"], cfg["features"]["label_col"])
    iv_path = Path(cfg["paths"]["iv_csv"])
    iv_path.parent.mkdir(parents=True, exist_ok=True)
    iv_summary.to_csv(iv_path, index=False)
    print(f"\nInformation Value (train split) -> {iv_path}")
    print(iv_summary.to_string(index=False))
