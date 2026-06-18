"""Config loading and results logging helpers shared across experiments."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML experiment config into a dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def log_results(results: Dict[str, Any], csv_path: str | Path) -> None:
    """Append a row of experiment results to a CSV, creating it if needed.

    Used so every baseline / GNN / temporal / autoencoder run lands in the
    same results table for cross-model comparison in the dissertation.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    row = {"timestamp": datetime.now(timezone.utc).isoformat(), **results}
    df = pd.DataFrame([row])

    if csv_path.exists():
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, mode="w", header=True, index=False)
