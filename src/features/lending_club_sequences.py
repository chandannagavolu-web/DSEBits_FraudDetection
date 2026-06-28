"""Per-loan temporal sequence construction for the Lending Club TCN.

The standard accepted-loans CSV is a one-row-per-loan **snapshot**, not a
monthly panel, so there is no genuine per-period repayment history to feed a
temporal model. This module's default ``synthetic_amortization`` mode therefore
manufactures a sequence from each loan's **amortization schedule**: given the
origination terms (principal, APR, term), it rolls the loan forward month by
month and records the scheduled balance / principal / interest trajectory. The
static origination features are (optionally) broadcast across every timestep so
the TCN has both the dynamic schedule and the cross-sectional risk signal.

This is an honest *proof-of-concept* temporal signal: the amortization path is
deterministic from the origination terms, so the discrimination still comes
mostly from the static features — but it exercises the full TCN training /
evaluation harness and is the drop-in slot for real sequences. When Lending
Club's payment-history panel (``PMTHIST_ALL_*.csv``) is available, implement the
``payment_history`` branch to replace the synthetic schedule with the actual
month-by-month ``RECEIVED`` / ``DUE`` / balance / delinquency records.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

# Dynamic (per-month) feature names, in column order.
DYNAMIC_FEATURES = [
    "bal_frac",        # outstanding balance / original principal
    "principal_frac",  # scheduled principal paid this month / principal
    "interest_frac",   # scheduled interest this month / principal
    "cum_paid_frac",   # cumulative principal repaid / principal
    "month_frac",      # month index / max_len (the temporal position)
]


def _amortization_tensor(
    principal: np.ndarray,
    apr_pct: np.ndarray,
    term_months: np.ndarray,
    max_len: int,
) -> np.ndarray:
    """Roll every loan's amortization schedule forward, vectorized over loans.

    Returns ``[n_loans, max_len, len(DYNAMIC_FEATURES)]``. Months beyond a loan's
    own term are zero-filled (the loan has closed).
    """
    n = len(principal)
    P = np.maximum(principal.astype(np.float64), 1.0)
    r = np.clip(apr_pct.astype(np.float64) / 100.0 / 12.0, 0.0, None)
    term = np.nan_to_num(term_months, nan=max_len).astype(np.float64)

    # Monthly payment M (annuity formula); r == 0 falls back to straight-line.
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = 1.0 - (1.0 + r) ** (-term)
        M = np.where(r > 0, P * r / np.where(denom == 0, np.nan, denom), P / np.maximum(term, 1.0))
    M = np.nan_to_num(M, nan=(P / np.maximum(term, 1.0)))

    seq = np.zeros((n, max_len, len(DYNAMIC_FEATURES)), dtype=np.float32)
    balance = P.copy()
    cum_paid = np.zeros(n, dtype=np.float64)

    for t in range(max_len):
        active = t < term
        interest_t = np.where(active, balance * r, 0.0)
        principal_t = np.where(active, np.minimum(M - interest_t, balance), 0.0)
        principal_t = np.clip(principal_t, 0.0, None)

        seq[:, t, 0] = (balance / P).astype(np.float32)
        seq[:, t, 1] = (principal_t / P).astype(np.float32)
        seq[:, t, 2] = (interest_t / P).astype(np.float32)
        cum_paid = cum_paid + principal_t
        seq[:, t, 3] = (cum_paid / P).astype(np.float32)
        seq[:, t, 4] = np.float32((t + 1) / max_len)

        balance = np.clip(balance - principal_t, 0.0, None)

    return seq


def build_sequences(
    raw_df: pd.DataFrame,
    static_feat_df: pd.DataFrame,
    static_cols: List[str],
    cfg: dict,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Build TCN input sequences for one split.

    Args:
        raw_df: cleaned loan rows (need ``loan_amnt``, ``int_rate``,
            ``term_months``, the label, and the id), index-aligned with
            ``static_feat_df``.
        static_feat_df: the processed feature frame (e.g. the scaled view) for
            the same loans.
        static_cols: which columns of ``static_feat_df`` to broadcast across
            timesteps (typically ``pipeline.scaled_feature_cols()``).
        cfg: full experiment config (uses ``sequences`` + ``features``).

    Returns ``(X, y, feature_names)`` where ``X`` is
    ``[n_loans, max_len, n_seq_features]`` float32.
    """
    seq_cfg = cfg["sequences"]
    label_col = cfg["features"]["label_col"]
    mode = seq_cfg.get("mode", "synthetic_amortization")
    max_len = int(seq_cfg["max_len"])

    if mode == "payment_history":
        raise NotImplementedError(
            "payment_history mode needs Lending Club's PMTHIST_ALL_*.csv panel; "
            "place it under data/raw/lending_club/ and implement the per-month "
            "RECEIVED/DUE/balance/delinquency sequence here. Use "
            "'synthetic_amortization' until then."
        )
    if mode != "synthetic_amortization":
        raise ValueError(f"Unknown sequences.mode: {mode!r}")

    raw_df = raw_df.sort_index()
    static_feat_df = static_feat_df.loc[raw_df.index]

    dyn = _amortization_tensor(
        principal=raw_df["loan_amnt"].to_numpy(),
        apr_pct=raw_df["int_rate"].to_numpy(),
        term_months=raw_df["term_months"].to_numpy(),
        max_len=max_len,
    )
    feature_names = list(DYNAMIC_FEATURES)

    if seq_cfg.get("static_in_sequence", True):
        static = static_feat_df[static_cols].to_numpy(dtype=np.float32)
        static_bcast = np.broadcast_to(static[:, None, :], (static.shape[0], max_len, static.shape[1]))
        dyn = np.concatenate([dyn, static_bcast], axis=2)
        feature_names += list(static_cols)

    y = raw_df[label_col].to_numpy(dtype=np.int64)
    return dyn.astype(np.float32), y, feature_names
