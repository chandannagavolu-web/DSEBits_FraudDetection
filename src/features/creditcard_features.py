"""Feature scaling for the Credit Card Fraud dataset (autoencoder track).

The features are already numeric (PCA components ``V1``..``V28`` plus ``Time`` /
``Amount``), so the pipeline is thin: drop ``Time`` if configured, then
standardize every retained feature. Crucially the scaler is fit on the
**train-normal** rows only — the autoencoder must learn the geometry of normal
traffic without the (rare) fraud cases or the held-out splits shifting the
mean/variance.

``transform`` returns a frame carrying ``TxnID`` + ``Class`` + the scaled
feature columns, so the same matrix feeds both the autoencoder and the
supervised baselines for a fair comparison.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.preprocessing import StandardScaler


class CreditCardFeaturePipeline:
    def __init__(self, label_col: str, id_col: str, drop_time: bool = True, standardize: bool = True) -> None:
        self.label_col = label_col
        self.id_col = id_col
        self.drop_time = drop_time
        self.standardize = standardize

        self.feature_cols_: List[str] = []
        self.scaler_: StandardScaler | None = None

    def _candidate_features(self, df: pd.DataFrame) -> List[str]:
        exclude = {self.label_col, self.id_col}
        if self.drop_time:
            exclude.add("Time")
        return [c for c in df.columns if c not in exclude]

    def fit(self, df: pd.DataFrame) -> "CreditCardFeaturePipeline":
        self.feature_cols_ = self._candidate_features(df)
        if self.standardize:
            self.scaler_ = StandardScaler().fit(df[self.feature_cols_].values)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.standardize and self.scaler_ is None:
            raise RuntimeError("Pipeline must be fit before calling transform().")

        out = pd.DataFrame(index=df.index)
        out[self.id_col] = df[self.id_col].values
        if self.label_col in df.columns:
            out[self.label_col] = df[self.label_col].values

        values = df[self.feature_cols_].values
        if self.standardize:
            values = self.scaler_.transform(values)
        for i, col in enumerate(self.feature_cols_):
            out[col] = values[:, i]
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def feature_cols(self) -> List[str]:
        return list(self.feature_cols_)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> "CreditCardFeaturePipeline":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_pipeline(cfg: dict) -> CreditCardFeaturePipeline:
    feat = cfg["features"]
    return CreditCardFeaturePipeline(
        label_col=feat["label_col"],
        id_col=feat["id_col"],
        drop_time=feat["drop_time"],
        standardize=feat["standardize"],
    )
