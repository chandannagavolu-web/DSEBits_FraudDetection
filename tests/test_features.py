import numpy as np
import pandas as pd

from src.features.ieee_cis_features import IEEECISFeaturePipeline


def make_synthetic_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "isFraud": rng.integers(0, 2, size=n),
            "TransactionAmt": rng.exponential(50, size=n),
            "ProductCD": rng.choice(["W", "C", "R", None], size=n),
            "card1": rng.integers(1000, 1010, size=n),
            "addr1": rng.choice([100.0, 200.0, np.nan], size=n),
            "P_emaildomain": rng.choice(["gmail.com", "yahoo.com", np.nan], size=n),
            "DeviceInfo": rng.choice(["Windows", "iOS", np.nan], size=n),
            "C1": rng.random(n),
            "D1": rng.choice([1.0, 2.0, np.nan], size=n),
        }
    )


def build_pipeline() -> IEEECISFeaturePipeline:
    return IEEECISFeaturePipeline(
        label_col="isFraud",
        id_col="TransactionID",
        categorical_cols=["ProductCD", "card1", "addr1", "P_emaildomain", "DeviceInfo"],
        freq_encode_cols=["card1", "addr1", "P_emaildomain", "DeviceInfo"],
        log1p_cols=["TransactionAmt"],
        missing_value=-999,
    )


def test_fit_transform_has_no_nans_and_consistent_columns():
    df = make_synthetic_df()
    pipeline = build_pipeline()

    feats = pipeline.fit_transform(df)

    tree_cols = pipeline.tree_feature_cols()
    scaled_cols = pipeline.scaled_feature_cols()

    assert not feats[tree_cols].isna().any().any()
    assert not feats[scaled_cols].isna().any().any()
    assert len(tree_cols) == len(scaled_cols)


def test_transform_handles_unseen_categories():
    df = make_synthetic_df()
    pipeline = IEEECISFeaturePipeline(
        label_col="isFraud",
        id_col="TransactionID",
        categorical_cols=["ProductCD"],
        freq_encode_cols=[],
        log1p_cols=["TransactionAmt"],
        missing_value=-999,
    )
    pipeline.fit(df)

    new_df = df.copy()
    new_df.loc[0, "ProductCD"] = "UNSEEN_CATEGORY"
    feats = pipeline.transform(new_df)

    unknown_idx = pipeline.label_encoders_["ProductCD"]["__unknown__"]
    assert feats.loc[0, "ProductCD"] == unknown_idx


def test_save_and_load_roundtrip(tmp_path):
    df = make_synthetic_df()
    pipeline = build_pipeline()
    pipeline.fit(df)

    path = tmp_path / "pipeline.pkl"
    pipeline.save(path)
    loaded = IEEECISFeaturePipeline.load(path)

    pd.testing.assert_frame_equal(pipeline.transform(df), loaded.transform(df))
