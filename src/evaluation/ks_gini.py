"""Credit-scoring KS / Gini evaluation utilities.

``src/evaluation/metrics.py`` already exposes the *continuous* KS (max gap of
the ROC tpr/fpr curves) and Gini (``2*AUC - 1``) that go into the shared
results table. This module adds the **decile / score-band** view that credit
risk teams actually report in a scorecard:

1. Rank all cases by predicted score (riskiest first).
2. Split into N equal-frequency bands (deciles by default).
3. Per band, accumulate the share of "bads" (positives / fraud) and "goods"
   (negatives) captured, and take KS = max separation between the two
   cumulative curves.

The decile KS is the discrimination number quoted against the industry rule of
thumb (KS > 0.40 => strong). It is computed from the empirical score
distribution rather than the full ROC sweep, so it can differ slightly from the
continuous KS in ``metrics.py`` — both are reported so the two are comparable.

The Gini here is the same ``2*AUC - 1`` rank-ordering measure, re-exported for
convenience so callers get the full credit view from one module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def ks_decile_table(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bands: int = 10,
) -> pd.DataFrame:
    """Build a credit-scorecard KS table over ``n_bands`` equal-frequency bands.

    Band 1 is the highest-risk (highest score) band. Ranking is done on a
    tie-broken rank rather than the raw score so the bands stay equal-sized even
    when a model emits many identical scores (e.g. a random forest pushing
    probabilities to exactly 0 / 1).

    Returns one row per band with counts, per-band bad rate, the cumulative
    share of bads and goods captured, and the running KS (in percentage points).
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    df = pd.DataFrame({"y": y_true, "score": y_score})
    total_bad = int(df["y"].sum())
    total_good = int(len(df) - total_bad)
    if total_bad == 0 or total_good == 0:
        raise ValueError("KS table needs both classes present in y_true.")

    # Tie-broken descending rank -> equal-frequency bands regardless of score ties.
    desc_rank = df["score"].rank(method="first", ascending=False)
    df["band"] = pd.qcut(desc_rank, n_bands, labels=False).astype(int) + 1

    grouped = df.groupby("band")
    table = pd.DataFrame(
        {
            "min_score": grouped["score"].min(),
            "max_score": grouped["score"].max(),
            "n": grouped["y"].size(),
            "n_bad": grouped["y"].sum(),
        }
    ).reset_index()
    table["n_good"] = table["n"] - table["n_bad"]
    table["bad_rate"] = table["n_bad"] / table["n"]

    table["cum_bad_pct"] = table["n_bad"].cumsum() / total_bad
    table["cum_good_pct"] = table["n_good"].cumsum() / total_good
    # KS in percentage points, matching how risk teams quote it (e.g. "KS = 43").
    table["ks"] = (table["cum_bad_pct"] - table["cum_good_pct"]).abs() * 100.0
    return table


def ks_from_table(table: pd.DataFrame) -> float:
    """Decile KS statistic (0-1 scale) = max band KS / 100."""
    return float(table["ks"].max() / 100.0)


def ks_continuous(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Continuous KS: max gap between tpr and fpr over the full ROC sweep."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(np.abs(tpr - fpr)))


def gini_coefficient(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Gini = 2 * AUC - 1."""
    return float(2 * roc_auc_score(y_true, y_score) - 1)


def ks_gini_summary(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bands: int = 10,
) -> Dict[str, float]:
    """One-row summary: AUC, Gini, decile KS, continuous KS, and which band the
    decile KS peaks in (useful to see how concentrated the separation is)."""
    table = ks_decile_table(y_true, y_score, n_bands=n_bands)
    auc = roc_auc_score(y_true, y_score)
    ks_decile = ks_from_table(table)
    return {
        "auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks_decile": ks_decile,
        "ks_continuous": ks_continuous(y_true, y_score),
        "ks_peak_band": int(table.loc[table["ks"].idxmax(), "band"]),
    }


def plot_ks_curve(table: pd.DataFrame, title: str, out_path: str | Path) -> None:
    """Save the cumulative bad vs good capture curves with the KS gap marked."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bands = table["band"].to_numpy()
    cum_bad = np.concatenate([[0.0], table["cum_bad_pct"].to_numpy()])
    cum_good = np.concatenate([[0.0], table["cum_good_pct"].to_numpy()])
    x = np.concatenate([[0], bands])

    ks_idx = int(table["ks"].idxmax())
    ks_band = int(table.loc[ks_idx, "band"])
    ks_val = float(table.loc[ks_idx, "ks"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, cum_bad * 100, marker="o", label="Cumulative % bad (fraud)")
    ax.plot(x, cum_good * 100, marker="s", label="Cumulative % good (legit)")
    ax.vlines(
        ks_band,
        table.loc[ks_idx, "cum_good_pct"] * 100,
        table.loc[ks_idx, "cum_bad_pct"] * 100,
        color="red",
        linestyles="--",
        label=f"KS = {ks_val:.1f} @ band {ks_band}",
    )
    ax.set_xlabel("Risk band (1 = highest score)")
    ax.set_ylabel("Cumulative population captured (%)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
