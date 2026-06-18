"""Reproducibility helpers."""

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (if available) RNGs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
