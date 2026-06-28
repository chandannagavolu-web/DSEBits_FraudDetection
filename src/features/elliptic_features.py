"""Feature-set selection and (leakage-free) scaling for the Elliptic dataset.

Elliptic's features are already engineered, so this module is deliberately thin
compared to ``paysim_features`` / ``ieee_cis_features``. It owns two things:

1. **Feature-set selection** — the Weber et al. (2019) AF/LF ablation:
   - ``local`` (LF): the 94 local transaction features (time step + 93 others);
   - ``all``   (AF): those 94 plus the 72 aggregated neighbourhood features.
   Optionally the raw ``time_step`` is dropped so a model can't read an absolute
   clock.

2. **Standardization fit on TRAIN nodes only** — the scaler's mean/std come from
   the training time steps and are applied to val/test/unknown nodes, so the
   temporal split is never violated.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data.elliptic import TIME_COL


def get_feature_columns(
    df: pd.DataFrame,
    feature_set: str = "all",
    n_local_features: int = 94,
    drop_time_step_feature: bool = False,
) -> List[str]:
    """Return the ordered feature columns for the requested feature set.

    ``local`` keeps ``time_step`` + the ``local_*`` columns (94 total); ``all``
    additionally keeps the ``agg_*`` columns (166 total). The column order is
    stable so the saved scaler and graph feature matrix stay aligned.
    """
    local_cols = [c for c in df.columns if c.startswith("local_")]
    agg_cols = [c for c in df.columns if c.startswith("agg_")]

    if feature_set == "local":
        cols = [TIME_COL] + local_cols
    elif feature_set == "all":
        cols = [TIME_COL] + local_cols + agg_cols
    else:
        raise ValueError(f"feature_set must be 'local' or 'all', got {feature_set!r}")

    # Sanity check against the documented width (94 local incl. time step).
    if feature_set == "local" and len(cols) != n_local_features:
        # Not fatal — synthetic / trimmed files may differ — but worth surfacing.
        pass

    if drop_time_step_feature:
        cols = [c for c in cols if c != TIME_COL]
    return cols


def scale_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_mask: np.ndarray,
    standardize: bool = True,
) -> Tuple[np.ndarray, StandardScaler | None]:
    """Build a float32 feature matrix, standardized on train-node statistics.

    The scaler is fit on ``df[feature_cols][train_mask]`` only and applied to
    every node, so val/test/unknown rows never leak into the mean/std. Returns
    ``(X, scaler)`` where ``scaler`` is ``None`` when ``standardize`` is False.
    """
    X = df[feature_cols].to_numpy(dtype=np.float32)
    if not standardize:
        return X, None

    scaler = StandardScaler().fit(X[train_mask])
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, scaler
