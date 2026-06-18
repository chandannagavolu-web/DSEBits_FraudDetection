"""SMOTE-based class-imbalance correction for the PaySim training split.

Run *after* ``src/features/paysim_features.py`` so it operates on the
fully-numeric feature table the baselines already use. Only the training
split is resampled — validation and test keep the original (~0.13%) fraud
rate so PR-AUC / MCC / KS / Gini reflect real-world performance.

PaySim is very imbalanced, so the default config keeps ``resampling.enabled:
false`` and relies on class-weighting / scale_pos_weight instead; this script
exists for the SMOTE-vs-classweight ablation.

Usage:
    python -m src.features.resampling_paysim
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.paysim_features import PaySimFeaturePipeline
from src.features.resampling import apply_smote
from src.utils.io import load_config
from src.utils.seed import set_seed

if __name__ == "__main__":
    cfg = load_config("configs/paysim.yaml")
    set_seed(cfg["seed"])

    processed_dir = Path(cfg["paths"]["processed_dir"])
    train = pd.read_parquet(processed_dir / "tabular_train.parquet")
    pipeline = PaySimFeaturePipeline.load(processed_dir / "feature_pipeline.pkl")

    label_col = cfg["features"]["label_col"]
    resample_cfg = cfg["resampling"]

    # Union of the tree (raw) and linear/GNN (scaled) feature views, so a
    # single resampled table serves both baseline families.
    feature_cols = sorted(set(pipeline.tree_feature_cols()) | set(pipeline.scaled_feature_cols()))

    before = train[label_col].value_counts().to_dict()
    resampled = apply_smote(
        train,
        feature_cols=feature_cols,
        label_col=label_col,
        sampling_strategy=resample_cfg["sampling_strategy"],
        k_neighbors=resample_cfg["k_neighbors"],
        random_state=cfg["seed"],
    )
    after = resampled[label_col].value_counts().to_dict()

    out_path = processed_dir / "tabular_train_smote.parquet"
    resampled.to_parquet(out_path)
    print(f"Class counts before SMOTE: {before}")
    print(f"Class counts after SMOTE:  {after}")
    print(f"Saved resampled training set -> {out_path}")
