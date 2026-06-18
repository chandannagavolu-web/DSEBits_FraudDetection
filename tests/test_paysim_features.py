import numpy as np
import pandas as pd

from src.features.paysim_features import PaySimFeaturePipeline, engineer_features


def make_synthetic_df(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(n),
            "step": rng.integers(1, 744, size=n),
            "type": rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], size=n),
            "amount": rng.exponential(1000, size=n),
            "nameOrig": [f"C{i % 20}" for i in range(n)],
            "oldbalanceOrg": rng.exponential(5000, size=n),
            "newbalanceOrig": rng.exponential(5000, size=n),
            "nameDest": rng.choice([f"M{i}" for i in range(10)] + [f"C{i}" for i in range(10)], size=n),
            "oldbalanceDest": rng.exponential(5000, size=n),
            "newbalanceDest": rng.exponential(5000, size=n),
            "isFraud": rng.integers(0, 2, size=n),
            "isFlaggedFraud": np.zeros(n, dtype=int),
        }
    )


def build_pipeline() -> PaySimFeaturePipeline:
    return PaySimFeaturePipeline(
        label_col="isFraud",
        id_col="TransactionID",
        categorical_cols=["type"],
        freq_encode_cols=["nameOrig", "nameDest"],
        log1p_cols=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"],
        drop_cols=["nameOrig", "nameDest", "isFlaggedFraud"],
        missing_value=-999,
    )


def test_engineer_features_balance_errors():
    df = make_synthetic_df(5)
    eng = engineer_features(df)
    expected_orig = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
    np.testing.assert_allclose(eng["errorBalanceOrig"].values, expected_orig.values)
    # merchant flag picks up the "M" prefix on nameDest
    assert set(eng["isMerchantDest"].unique()) <= {0, 1}


def test_fit_transform_has_no_nans_and_consistent_columns():
    df = make_synthetic_df()
    pipeline = build_pipeline()

    feats = pipeline.fit_transform(df)

    tree_cols = pipeline.tree_feature_cols()
    scaled_cols = pipeline.scaled_feature_cols()

    assert not feats[tree_cols].isna().any().any()
    assert not feats[scaled_cols].isna().any().any()
    assert len(tree_cols) == len(scaled_cols)


def test_leaky_and_id_columns_excluded_from_features():
    df = make_synthetic_df()
    pipeline = build_pipeline()
    pipeline.fit(df)

    # raw account ids and isFlaggedFraud must not become numeric features
    for dropped in ["nameOrig", "nameDest", "isFlaggedFraud"]:
        assert dropped not in pipeline.numeric_cols_
    # but account ids are still frequency-encoded
    assert "nameOrig_freq" in pipeline.tree_feature_cols()
    assert "nameDest_freq" in pipeline.tree_feature_cols()


def test_transform_handles_unseen_type():
    df = make_synthetic_df()
    pipeline = build_pipeline()
    pipeline.fit(df)

    new_df = df.copy()
    new_df.loc[0, "type"] = "UNSEEN_TYPE"
    feats = pipeline.transform(new_df)

    unknown_idx = pipeline.label_encoders_["type"]["__unknown__"]
    assert feats.loc[0, "type"] == unknown_idx


def test_save_and_load_roundtrip(tmp_path):
    df = make_synthetic_df()
    pipeline = build_pipeline()
    pipeline.fit(df)

    path = tmp_path / "pipeline.pkl"
    pipeline.save(path)
    loaded = PaySimFeaturePipeline.load(path)

    pd.testing.assert_frame_equal(pipeline.transform(df), loaded.transform(df))
