"""End-to-end Credit Card Fraud pipeline (autoencoder anomaly-detection track).

load -> features -> supervised baselines -> autoencoder -> AE analysis.

The feature scaler is fit on the **train-normal** rows only, so the autoencoder
learns normal-traffic geometry without fraud (or held-out splits) shifting the
scale. Assumes the raw ``creditcard.csv`` is under ``data/raw/creditcard/``
(see ``data/README.md``); it does not download anything.

Usage:
    python run_creditcard_pipeline.py
"""

from pathlib import Path

from src.data.creditcard import load_and_split, normal_only, save_interim
from src.evaluation.evaluate_creditcard import run as evaluate_ae
from src.features.creditcard_features import (
    CreditCardFeaturePipeline,  # noqa: F401  (dotted-path import for stable pickling)
    build_pipeline,
)
from src.models.autoencoder.train_autoencoder_creditcard import run as train_ae
from src.models.baseline.train_baseline_creditcard import run as train_baselines
from src.utils.io import load_config

cfg = load_config("configs/creditcard.yaml")

# 1. Load + stratified split.
df, splits = load_and_split(cfg)
save_interim(df, cfg["paths"]["interim_dir"])
print(f"Loaded {len(df)} transactions, fraud rate = {df['Class'].mean():.4%}")
for name, sdf in splits.items():
    print(f"  {name}: {len(sdf)} rows, fraud rate = {sdf['Class'].mean():.4%}")

# 2. Fit the scaler on TRAIN-NORMAL only, transform every split.
label_col = cfg["features"]["label_col"]
pipeline = build_pipeline(cfg)
pipeline.fit(normal_only(splits["train"], label_col))
processed = {name: pipeline.transform(sdf) for name, sdf in splits.items()}

processed_dir = Path(cfg["paths"]["processed_dir"])
processed_dir.mkdir(parents=True, exist_ok=True)
for name, frame in processed.items():
    frame.to_parquet(processed_dir / f"tabular_{name}.parquet")
pipeline.save(processed_dir / "feature_pipeline.pkl")
print(f"feature matrix: {len(pipeline.feature_cols())} features "
      f"(scaler fit on {len(normal_only(splits['train'], label_col))} normal train rows)")

# 3. Supervised baselines (Objective 4 comparison).
print("\n=== supervised baselines ===")
train_baselines(cfg)

# 4. Autoencoder anomaly detector (Objective 3).
print("\n=== autoencoder ===")
train_ae(cfg)

# 5. AE analysis (error distribution, PR curve, threshold sweep).
print("\n=== autoencoder analysis ===")
evaluate_ae(cfg)

print("\nCompleted Credit Card Fraud workflow")
