"""Reproducibility utilities for PFGT-UIE.

Implements seed_everything() which sets all known random state sources to a
fixed seed so that training runs are deterministic and reproducible.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy, PyTorch (CPU + CUDA) and configure cuDNN for reproducibility.

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # cuDNN deterministic mode: slower but reproducible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
