"""PSNR (Peak Signal-to-Noise Ratio) metric for image quality assessment.

Standard definition: PSNR = 10 * log10(MAX^2 / MSE)
For [0, 1] normalized images, MAX = 1.0.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def compute_psnr(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    data_range: float = 1.0,
) -> float:
    """Compute PSNR between a predicted image and a target image.

    Args:
        prediction: Predicted image. Shape (B, C, H, W), (C, H, W), or (H, W).
                    Values expected in [0, data_range].
        target:     Ground-truth image. Same shape as prediction.
        data_range: Maximum possible pixel value (1.0 for normalized images).

    Returns:
        Average PSNR in dB over the batch. Returns inf if MSE == 0.
    """
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().float().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().float().numpy()

    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, "
            f"got {prediction.shape} and {target.shape}."
        )

    # Flatten to (N, -1) where N is the batch size (or 1 if no batch dim)
    if prediction.ndim == 2:
        prediction = prediction[np.newaxis, np.newaxis]
        target = target[np.newaxis, np.newaxis]
    elif prediction.ndim == 3:
        prediction = prediction[np.newaxis]
        target = target[np.newaxis]

    batch_size = prediction.shape[0]
    psnr_values: list[float] = []

    for i in range(batch_size):
        mse = np.mean((prediction[i] - target[i]) ** 2)
        if mse == 0.0:
            psnr_values.append(float("inf"))
        else:
            psnr_values.append(10.0 * math.log10((data_range**2) / mse))

    return float(np.mean(psnr_values))
