"""Unit tests for the Lending Club credit-risk track.

Focus on the three things that make this track easy to get subtly wrong:
leakage control, the terminal-status target definition, and the out-of-time
split ordering — plus the feature pipeline and the TCN sequence builder.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.lending_club import (
    ID_COL,
    _clean_chunk,
    _parse_emp_length,
    _parse_month_year,
    _parse_percent,
    _parse_term,
    make_temporal_splits,
)
from src.features.lending_club_features import LendingClubFeaturePipeline, build_pipeline
from src.features.lending_club_sequences import DYNAMIC_FEATURES, build_sequences

# --- leakage allowlist: anything matching these must never reach a model ------
LEAKAGE_SUBSTRINGS = [
    "recoveries", "total_pymnt", "total_rec", "last_pymnt", "next_pymnt",
    "out_prncp", "last_fico", "hardship", "settlement", "collection_recovery",
]

TARGET_CFG = {
    "bad_statuses": ["Charged Off", "Default", "Does not meet the credit policy. Status:Charged Off"],
    "good_statuses": ["Fully Paid", "Does not meet the credit policy. Status:Fully Paid"],
}

NUMERIC_COLS = [
    "loan_amnt", "term_months", "int_rate", "installment", "emp_length_years",
    "annual_inc", "dti", "delinq_2yrs", "fico", "credit_age_mths",
    "inq_last_6mths", "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
]
CATEGORICAL_COLS = [
    "grade", "sub_grade", "home_ownership", "verification_status",
    "purpose", "initial_list_status", "application_type",
]


def make_cfg() -> dict:
    return {
        "features": {
            "label_col": "target", "id_col": ID_COL,
            "numeric_cols": NUMERIC_COLS, "categorical_cols": CATEGORICAL_COLS,
            "freq_encode_cols": ["addr_state"],
            "log1p_cols": ["annual_inc", "revol_bal", "installment"],
            "missing_value": -999,
        },
        "sequences": {"mode": "synthetic_amortization", "max_len": 36, "static_in_sequence": True},
    }


def make_raw_chunk(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    statuses = rng.choice(
        ["Fully Paid", "Charged Off", "Current", "In Grace Period", "Default"],
        size=n, p=[0.5, 0.2, 0.15, 0.1, 0.05],
    )
    issue = pd.to_datetime("2013-01-01") + pd.to_timedelta(rng.integers(0, 1800, n), unit="D")
    earliest = issue - pd.to_timedelta(rng.integers(400, 5000, n), unit="D")
    return pd.DataFrame({
        "loan_status": statuses,
        "issue_d": issue.strftime("%b-%Y"),
        "earliest_cr_line": earliest.strftime("%b-%Y"),
        "term": rng.choice(["36 months", "60 months"], size=n),
        "emp_length": rng.choice(["< 1 year", "3 years", "10+ years", "n/a"], size=n),
        "int_rate": [f"{x:.2f}%" for x in rng.uniform(6, 28, n)],
        "revol_util": [f"{x:.1f}%" for x in rng.uniform(0, 100, n)],
        "fico_range_low": rng.integers(660, 800, n),
        "fico_range_high": rng.integers(660, 800, n) + 4,
        "loan_amnt": rng.integers(1000, 35000, n).astype(float),
        "installment": rng.uniform(30, 1200, n),
        "annual_inc": rng.uniform(20000, 200000, n),
        "dti": rng.uniform(0, 35, n),
        "delinq_2yrs": rng.integers(0, 3, n),
        "inq_last_6mths": rng.integers(0, 4, n),
        "open_acc": rng.integers(2, 25, n),
        "pub_rec": rng.integers(0, 2, n),
        "revol_bal": rng.integers(0, 50000, n),
        "total_acc": rng.integers(3, 50, n),
        "grade": rng.choice(list("ABCDEF"), size=n),
        "sub_grade": rng.choice(["A1", "B3", "C2", "D5"], size=n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], size=n),
        "verification_status": rng.choice(["Verified", "Not Verified"], size=n),
        "purpose": rng.choice(["debt_consolidation", "credit_card", "other"], size=n),
        "addr_state": rng.choice(["CA", "TX", "NY", "FL"], size=n),
        "initial_list_status": rng.choice(["w", "f"], size=n),
        "application_type": rng.choice(["Individual", "Joint App"], size=n),
    })


def make_clean_df(n: int = 300) -> pd.DataFrame:
    """A post-loader cleaned dataframe (what splits/pipeline/sequences see)."""
    chunk = make_raw_chunk(n)
    df = _clean_chunk(chunk, TARGET_CFG)
    df.insert(0, ID_COL, range(len(df)))
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- parsing -----
def test_parsers():
    assert _parse_term(pd.Series(["36 months", " 60 months"])).tolist() == [36, 60]
    el = _parse_emp_length(pd.Series(["10+ years", "< 1 year", "3 years", "n/a"]))
    assert el.tolist()[:3] == [10, 0, 3]
    assert np.isnan(el.tolist()[3])
    assert _parse_percent(pd.Series(["13.56%", "7%"])).round(2).tolist() == [13.56, 7.0]
    assert _parse_month_year(pd.Series(["Dec-2015"]))[0] == pd.Timestamp("2015-12-01")


# ------------------------------------------------------- target / censoring ---
def test_target_keeps_only_terminal_status_and_maps_bad_to_one():
    df = _clean_chunk(make_raw_chunk(400), TARGET_CFG)
    # No in-progress statuses survived.
    assert set(df["loan_status"].unique()).issubset(set(TARGET_CFG["bad_statuses"]) | set(TARGET_CFG["good_statuses"]))
    # bad = 1 exactly for the charge-off / default statuses.
    bad_mask = df["loan_status"].isin(TARGET_CFG["bad_statuses"])
    assert (df.loc[bad_mask, "target"] == 1).all()
    assert (df.loc[~bad_mask, "target"] == 0).all()


def test_clean_engineers_expected_columns():
    df = _clean_chunk(make_raw_chunk(50), TARGET_CFG)
    for col in ["target", "issue_date", "credit_age_mths", "term_months", "emp_length_years", "fico"]:
        assert col in df.columns
    assert (df["credit_age_mths"] >= 0).all()


# ------------------------------------------------------------- leakage --------
def test_no_leakage_columns_in_feature_matrix():
    df = make_clean_df()
    feats = build_pipeline(make_cfg()).fit_transform(df)
    for col in feats.columns:
        low = col.lower()
        assert not any(s in low for s in LEAKAGE_SUBSTRINGS), f"leakage column leaked: {col}"


# --------------------------------------------------- out-of-time split --------
def test_temporal_split_is_strictly_ordered_in_time():
    df = make_clean_df(300)
    splits = make_temporal_splits(df, date_col="issue_date", train_frac=0.7, val_frac=0.15, test_frac=0.15)
    # every train loan issued no later than every val loan, etc. (out-of-time)
    assert splits["train"]["issue_date"].max() <= splits["val"]["issue_date"].min()
    assert splits["val"]["issue_date"].max() <= splits["test"]["issue_date"].min()
    # partition covers all rows with no overlap
    total = sum(len(v) for v in splits.values())
    assert total == len(df)
    ids = pd.concat([v[ID_COL] for v in splits.values()])
    assert ids.is_unique


def test_temporal_split_rejects_bad_fractions():
    df = make_clean_df(50)
    with pytest.raises(ValueError):
        make_temporal_splits(df, train_frac=0.6, val_frac=0.2, test_frac=0.1)


# ------------------------------------------------------ feature pipeline ------
def test_pipeline_no_nans_and_freq_encoding_and_roundtrip(tmp_path):
    df = make_clean_df()
    pipe = build_pipeline(make_cfg())
    feats = pipe.fit_transform(df)

    assert not feats[pipe.tree_feature_cols()].isna().any().any()
    assert not feats[pipe.scaled_feature_cols()].isna().any().any()
    assert "addr_state_freq" in feats.columns  # high-card column frequency-encoded
    assert len(pipe.tree_feature_cols()) == len(pipe.scaled_feature_cols())

    path = tmp_path / "pipe.pkl"
    pipe.save(path)
    pd.testing.assert_frame_equal(feats, LendingClubFeaturePipeline.load(path).transform(df))


def test_pipeline_handles_unseen_category():
    df = make_clean_df()
    pipe = build_pipeline(make_cfg())
    pipe.fit(df)
    new = df.copy()
    new.loc[new.index[0], "purpose"] = "moon_base"
    out = pipe.transform(new)
    assert out.iloc[0]["purpose"] == pipe.label_encoders_["purpose"]["__unknown__"]


# ------------------------------------------------------- TCN sequences --------
def test_build_sequences_shapes_and_amortization_is_causal():
    df = make_clean_df(120)
    cfg = make_cfg()
    pipe = build_pipeline(cfg)
    feats = pipe.fit_transform(df)

    X, y, names = build_sequences(df, feats, pipe.scaled_feature_cols(), cfg)
    n, T, F = X.shape
    assert T == cfg["sequences"]["max_len"]
    assert n == len(df) == len(y)
    # dynamic features first, then broadcast static features
    assert names[: len(DYNAMIC_FEATURES)] == DYNAMIC_FEATURES
    assert F == len(DYNAMIC_FEATURES) + len(pipe.scaled_feature_cols())

    # Balance fraction (feature 0) must be non-increasing over time (a loan only
    # ever pays down) and start at ~1.0.
    bal = X[:, :, 0]
    assert np.allclose(bal[:, 0], 1.0, atol=1e-3)
    assert (np.diff(bal, axis=1) <= 1e-5).all()


def test_static_broadcast_is_constant_across_time():
    df = make_clean_df(40)
    cfg = make_cfg()
    pipe = build_pipeline(cfg)
    feats = pipe.fit_transform(df)
    X, _, _ = build_sequences(df, feats, pipe.scaled_feature_cols(), cfg)
    # first static channel (index len(DYNAMIC_FEATURES)) is identical at every step
    static_chan = X[:, :, len(DYNAMIC_FEATURES)]
    assert np.allclose(static_chan, static_chan[:, :1])


def test_payment_history_mode_not_implemented():
    df = make_clean_df(10)
    cfg = make_cfg()
    cfg["sequences"]["mode"] = "payment_history"
    pipe = build_pipeline(cfg)
    feats = pipe.fit_transform(df)
    with pytest.raises(NotImplementedError):
        build_sequences(df, feats, pipe.scaled_feature_cols(), cfg)
