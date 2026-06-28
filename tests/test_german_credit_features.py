import numpy as np
import pandas as pd
import pytest

from src.data.german_credit import _normalize_label, load_raw
from src.features.german_credit_features import (
    GermanCreditFeaturePipeline,
    compute_woe_iv,
    information_value_summary,
)


def make_synthetic_df(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "ApplicantID": np.arange(n),
            "Age": rng.integers(19, 75, size=n),
            "Sex": rng.choice(["male", "female"], size=n),
            "Job": rng.integers(0, 4, size=n),
            "Housing": rng.choice(["own", "rent", "free"], size=n),
            "Saving accounts": rng.choice(["little", "moderate", "rich", None], size=n),
            "Checking account": rng.choice(["little", "moderate", None], size=n),
            "Credit amount": rng.integers(250, 18000, size=n),
            "Duration": rng.integers(4, 72, size=n),
            "Purpose": rng.choice(["car", "radio/TV", "education", "business"], size=n),
            "Risk": rng.integers(0, 2, size=n),
        }
    )
    return df


def build_pipeline() -> GermanCreditFeaturePipeline:
    return GermanCreditFeaturePipeline(
        label_col="Risk",
        id_col="ApplicantID",
        categorical_cols=[
            "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Purpose",
        ],
        log1p_cols=["Credit amount"],
        missing_value=-999,
    )


def test_normalize_label_handles_all_encodings():
    # cleaned good/bad strings -> bad = 1
    assert _normalize_label(pd.Series(["good", "bad", "good"])).tolist() == [0, 1, 0]
    # UCI Statlog 1/2 -> 2 (bad) becomes 1
    assert _normalize_label(pd.Series([1, 2, 1])).tolist() == [0, 1, 0]
    # already 0/1 is unchanged
    assert _normalize_label(pd.Series([0, 1, 1])).tolist() == [0, 1, 1]


def test_load_raw_requires_a_target_column_with_helpful_error(tmp_path):
    feature_only = pd.DataFrame({"Age": [20, 30], "Sex": ["male", "female"]})
    path = tmp_path / "german_credit_data.csv"
    feature_only.to_csv(path, index=False)

    with pytest.raises(ValueError, match="No target/label column found"):
        load_raw(tmp_path)


def test_fit_transform_has_no_nans_and_consistent_columns():
    df = make_synthetic_df()
    pipeline = build_pipeline()
    feats = pipeline.fit_transform(df)

    tree_cols = pipeline.tree_feature_cols()
    scaled_cols = pipeline.scaled_feature_cols()

    # NaN-bearing account columns must be encoded, not left as NaN.
    assert not feats[tree_cols].isna().any().any()
    assert not feats[scaled_cols].isna().any().any()
    assert len(tree_cols) == len(scaled_cols)


def test_id_and_label_excluded_from_features():
    df = make_synthetic_df()
    pipeline = build_pipeline()
    pipeline.fit(df)

    for excluded in ["ApplicantID", "Risk"]:
        assert excluded not in pipeline.numeric_cols_
        assert excluded not in pipeline.tree_feature_cols()


def test_transform_handles_unseen_category():
    df = make_synthetic_df()
    pipeline = build_pipeline()
    pipeline.fit(df)

    new_df = df.copy()
    new_df.loc[0, "Purpose"] = "spaceship"
    feats = pipeline.transform(new_df)

    unknown_idx = pipeline.label_encoders_["Purpose"]["__unknown__"]
    assert feats.loc[0, "Purpose"] == unknown_idx


def test_missing_category_encoded_not_dropped():
    df = make_synthetic_df()
    pipeline = build_pipeline()
    pipeline.fit(df)
    # the explicit "missing" category exists for the NaN-bearing column
    assert "missing" in pipeline.label_encoders_["Saving accounts"]


def test_save_and_load_roundtrip(tmp_path):
    df = make_synthetic_df()
    pipeline = build_pipeline()
    pipeline.fit(df)

    path = tmp_path / "pipeline.pkl"
    pipeline.save(path)
    loaded = GermanCreditFeaturePipeline.load(path)

    pd.testing.assert_frame_equal(pipeline.transform(df), loaded.transform(df))


def test_information_value_is_finite_and_ranked():
    df = make_synthetic_df()
    summary = information_value_summary(
        df, ["Age", "Credit amount", "Duration", "Purpose"], "Risk"
    )
    assert np.isfinite(summary["iv"]).all()
    # sorted strongest-first
    assert summary["iv"].is_monotonic_decreasing
    assert set(summary["strength"]).issubset(
        {"useless", "weak", "medium", "strong", "suspicious"}
    )


def test_woe_iv_separates_a_perfectly_predictive_feature():
    # A feature that perfectly tracks the label should yield a high IV.
    n = 200
    y = np.array([0, 1] * (n // 2))
    df = pd.DataFrame({"signal": y.astype(str), "Risk": y})
    table = compute_woe_iv(df, "signal", "Risk")
    assert float(table["iv_contrib"].sum()) > 0.5
