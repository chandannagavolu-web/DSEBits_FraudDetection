"""End-to-end Lending Club pipeline: load -> features -> baselines -> KS/Gini -> TCN.

Mirrors ``run_german_credit_pipeline.py`` but for the large, real-world
credit-risk dataset, adding the temporal TCN stage (Phase 2). Assumes the raw
``accepted_2007_to_2018Q4.csv`` is already under ``data/raw/lending_club/``
(see ``data/README.md``); it does not download anything.

Usage:
    python run_lending_club_pipeline.py
"""

from pathlib import Path

from src.data.lending_club import load_and_split, save_interim
from src.evaluation.evaluate_lending_club_ks_gini import run as evaluate_ks_gini
from src.features.lending_club_features import (
    LendingClubFeaturePipeline,  # noqa: F401  (dotted-path import for stable pickling)
    build_pipeline,
    information_value_summary,
)
from src.models.baseline.train_baseline_lending_club import run as train_baselines
from src.models.temporal.train_tcn_lending_club import run as train_tcn
from src.utils.io import load_config

cfg = load_config("configs/lending_club.yaml")

# 1. Load + clean + out-of-time split.
df, splits = load_and_split(cfg)
save_interim(df, cfg["paths"]["interim_dir"])
print(f"Loaded {len(df)} terminal-status loans; vintages "
      f"{df['issue_date'].min():%Y-%m}..{df['issue_date'].max():%Y-%m}")
for name, split_df in splits.items():
    print(f"  {name}: {len(split_df)} loans, bad rate = {split_df['target'].mean():.4%}")

# 2. Fit the feature pipeline on train, transform all splits, persist.
pipeline = build_pipeline(cfg)
train_feats = pipeline.fit_transform(splits["train"])
val_feats = pipeline.transform(splits["val"])
test_feats = pipeline.transform(splits["test"])
print("feature shapes", train_feats.shape, val_feats.shape, test_feats.shape)

processed_dir = Path(cfg["paths"]["processed_dir"])
processed_dir.mkdir(parents=True, exist_ok=True)
for name, feats in [("train", train_feats), ("val", val_feats), ("test", test_feats)]:
    feats.to_parquet(processed_dir / f"tabular_{name}.parquet")
pipeline.save(processed_dir / "feature_pipeline.pkl")

# 3. WoE / Information Value screening (train split only).
iv_summary = information_value_summary(splits["train"], cfg["features"]["iv_cols"], cfg["features"]["label_col"])
iv_path = Path(cfg["paths"]["iv_csv"])
iv_path.parent.mkdir(parents=True, exist_ok=True)
iv_summary.to_csv(iv_path, index=False)
print("\nTop Information Value features:")
print(iv_summary.head().to_string(index=False))

# 4. Tabular baselines (Phase 1) + KS/Gini scorecard.
print("\n=== baselines ===")
train_baselines(cfg)
print("\n=== KS/Gini scorecard ===")
evaluate_ks_gini(cfg)

# 5. Temporal TCN (Phase 2).
print("\n=== TCN ===")
train_tcn(cfg)

print("\nCompleted Lending Club workflow")
