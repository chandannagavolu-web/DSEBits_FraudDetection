"""Unit tests for the Credit Card Fraud (autoencoder) track."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.creditcard import ID_COL, load_raw, make_splits, normal_only
from src.features.creditcard_features import CreditCardFeaturePipeline, build_pipeline
from src.models.autoencoder.autoencoder import Autoencoder
from src.models.autoencoder.train_autoencoder_creditcard import choose_threshold


def make_df(n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {"Time": rng.uniform(0, 1e5, n), "Amount": rng.exponential(80, n)}
    for i in range(1, 29):
        data[f"V{i}"] = rng.normal(size=n)
    data["Class"] = (rng.random(n) < 0.1).astype(int)
    df = pd.DataFrame(data)
    df.insert(0, ID_COL, range(n))
    return df


CFG = {
    "features": {"label_col": "Class", "id_col": ID_COL, "drop_time": True, "standardize": True}
}


# ----------------------------------------------------------------- data -------
def test_load_raw_requires_class(tmp_path):
    pd.DataFrame({"V1": [1.0, 2.0], "Amount": [3.0, 4.0]}).to_csv(tmp_path / "creditcard.csv", index=False)
    with pytest.raises(ValueError, match="Class"):
        load_raw(tmp_path)


def test_load_raw_adds_id(tmp_path):
    df = make_df(20).drop(columns=[ID_COL])
    df.to_csv(tmp_path / "creditcard.csv", index=False)
    loaded = load_raw(tmp_path)
    assert ID_COL in loaded.columns and loaded[ID_COL].is_unique


def test_splits_are_stratified_and_partition():
    df = make_df()
    splits = make_splits(df, label_col="Class")
    assert sum(len(v) for v in splits.values()) == len(df)
    rates = [v["Class"].mean() for v in splits.values()]
    assert max(rates) - min(rates) < 0.08  # roughly preserved fraud rate
    assert (normal_only(splits["train"])["Class"] == 0).all()


# ------------------------------------------------------------ features --------
def test_pipeline_drops_time_scales_and_roundtrips(tmp_path):
    df = make_df()
    pipe = build_pipeline(CFG)
    feats = pipe.fit(normal_only(df)).transform(df)

    assert "Time" not in pipe.feature_cols()
    assert len(pipe.feature_cols()) == 29  # V1..V28 + Amount
    assert not feats[pipe.feature_cols()].isna().any().any()
    # scaler was fit on normal rows -> their scaled features are ~zero-mean.
    normal_scaled = feats.loc[df["Class"] == 0, pipe.feature_cols()].to_numpy()
    assert np.allclose(normal_scaled.mean(axis=0), 0, atol=1e-6)

    path = tmp_path / "p.pkl"
    pipe.save(path)
    pd.testing.assert_frame_equal(feats, CreditCardFeaturePipeline.load(path).transform(df))


# ----------------------------------------------------------- autoencoder ------
def test_autoencoder_shapes_and_bottleneck():
    ae = Autoencoder(in_dim=29, encoder_dims=[20, 14, 7])
    x = torch.randn(16, 29)
    assert ae(x).shape == (16, 29)
    assert ae.encoder(x).shape == (16, 7)  # bottleneck width
    assert ae.reconstruction_error(x).shape == (16,)


def test_autoencoder_flags_anomalies_after_training_on_normal():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    # normal: tight gaussian; anomalies: shifted/scaled -> should reconstruct worse
    normal = rng.normal(0, 1, size=(2000, 8)).astype(np.float32)
    anom = rng.normal(6, 1, size=(200, 8)).astype(np.float32)

    ae = Autoencoder(in_dim=8, encoder_dims=[6, 3])
    opt = torch.optim.Adam(ae.parameters(), lr=0.01)
    xb = torch.from_numpy(normal)
    for _ in range(150):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(ae(xb), xb)
        loss.backward()
        opt.step()

    err_normal = ae.reconstruction_error(torch.from_numpy(normal)).mean().item()
    err_anom = ae.reconstruction_error(torch.from_numpy(anom)).mean().item()
    assert err_anom > err_normal * 2  # anomalies reconstruct clearly worse


def test_choose_threshold_methods():
    y = np.array([0, 0, 0, 1, 1])
    err = np.array([0.1, 0.2, 0.15, 0.9, 0.8])
    cfg_f1 = {"autoencoder": {"threshold_method": "best_f1"}}
    cfg_pct = {"autoencoder": {"threshold_method": "percentile", "threshold_percentile": 90.0}}
    t1 = choose_threshold(y, err, cfg_f1)
    t2 = choose_threshold(y, err, cfg_pct)
    assert np.isfinite(t1) and np.isfinite(t2)
    # a good F1 threshold separates the two fraud points (err ~0.8-0.9) from normal
    assert 0.2 <= t1 <= 0.9
