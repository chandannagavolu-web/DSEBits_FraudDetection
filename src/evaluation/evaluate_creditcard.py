"""Autoencoder anomaly-detection analysis for the Credit Card Fraud track.

The training script logs the headline metrics; this module adds the views that
make a reconstruction-error detector interpretable:

- the **reconstruction-error distribution** for normal vs fraud transactions
  (the visual that justifies thresholding the error at all)
  -> reports/figures/creditcard_recon_error_dist.png
- the **precision-recall curve** of the error score on the test split
  -> reports/figures/creditcard_pr_curve.png
- a **threshold sweep** table (precision / recall / F1 vs error threshold)
  -> reports/creditcard/threshold_sweep.csv

Run after training the autoencoder:
    python -m src.evaluation.evaluate_creditcard
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.models.autoencoder.autoencoder import Autoencoder
from src.models.autoencoder.train_autoencoder_creditcard import _errors, load_processed
from src.utils.io import load_config
from src.utils.seed import set_seed


def _load_autoencoder(cfg: dict, in_dim: int) -> tuple[Autoencoder, float]:
    model_dir = Path(cfg["paths"]["model_dir"])
    with open(model_dir / "autoencoder_meta.json") as f:
        meta = json.load(f)
    model = Autoencoder(
        in_dim=in_dim,
        encoder_dims=meta["encoder_dims"],
        dropout=cfg["autoencoder"]["dropout"],
        activation=cfg["autoencoder"]["activation"],
    )
    model.load_state_dict(torch.load(model_dir / "autoencoder.pt", weights_only=True))
    model.eval()
    return model, float(meta["threshold"])


def run(cfg: dict) -> None:
    set_seed(cfg["seed"])

    _, _, test, pipeline = load_processed(cfg["paths"]["processed_dir"])
    cols = pipeline.feature_cols()
    label_col = cfg["features"]["label_col"]

    model, threshold = _load_autoencoder(cfg, in_dim=len(cols))
    X_test = test[cols].to_numpy(dtype=np.float32)
    y_test = test[label_col].to_numpy()
    err = _errors(model, X_test)

    out_dir = Path(cfg["paths"]["results_csv"]).parent / "creditcard"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("reports/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ---- threshold sweep -----------------------------------------------------
    precision, recall, thr = precision_recall_curve(y_test, err)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    sweep = pd.DataFrame({
        "threshold": np.append(thr, np.nan),
        "precision": precision, "recall": recall, "f1": f1,
    })
    sweep_path = out_dir / "threshold_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    ap = average_precision_score(y_test, err)
    chosen_f1 = float(f1[:-1][np.argmin(np.abs(thr - threshold))]) if len(thr) else float("nan")
    print(f"AE test PR-AUC (avg precision) = {ap:.4f}")
    print(f"Chosen threshold = {threshold:.6f} -> F1 ~ {chosen_f1:.4f}")
    print(f"Saved threshold sweep -> {sweep_path}")

    _plot_error_distribution(err, y_test, threshold, fig_dir / "creditcard_recon_error_dist.png")
    _plot_pr_curve(recall, precision, ap, fig_dir / "creditcard_pr_curve.png")
    print(f"Saved figures -> {fig_dir}")


def _plot_error_distribution(err, y, threshold, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, np.quantile(err, 0.999), 80)
    ax.hist(err[y == 0], bins=bins, alpha=0.6, label="normal", color="steelblue", density=True)
    ax.hist(err[y == 1], bins=bins, alpha=0.6, label="fraud", color="crimson", density=True)
    ax.axvline(threshold, color="black", ls="--", lw=1.2, label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("Reconstruction error (MSE)")
    ax.set_ylabel("Density")
    ax.set_title("Credit Card — reconstruction error: normal vs fraud")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def _plot_pr_curve(recall, precision, ap, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall, precision, lw=2, label=f"AE (PR-AUC = {ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Credit Card — autoencoder precision-recall curve (test)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    run(load_config("configs/creditcard.yaml"))
