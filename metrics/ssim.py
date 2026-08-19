"""SSIM (Structural Similarity Index) metric for image quality assessment.

Implements the standard SSIM formulation from:
Wang et al. (2004). "Image quality assessment: from error visibility to structural similarity."
IEEE Transactions on Image Processing.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_ssim(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    data_range: float = 1.0,
    kernel_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Compute mean SSIM between predicted and target images.

    Args:
        prediction:  Predicted image tensor. Shape (B, C, H, W) or (C, H, W).
        target:      Ground-truth image tensor. Same shape as prediction.
        data_range:  Max pixel value (1.0 for normalized images).
        kernel_size: Gaussian kernel size (odd). Default 11 follows the paper.
        sigma:       Gaussian kernel standard deviation. Default 1.5 follows the paper.

    Returns:
        Average SSIM value over the batch (scalar float).
    """
    if isinstance(prediction, np.ndarray):
        prediction = torch.from_numpy(prediction).float()
    if isinstance(target, np.ndarray):
        target = torch.from_numpy(target).float()

    prediction = prediction.float()
    target = target.float()

    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
        target = target.unsqueeze(0)

    # Build Gaussian kernel on the same device as inputs
    kernel = _gaussian_kernel(kernel_size, sigma, channels=prediction.shape[1]).to(
        prediction.device
    )

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    padding = kernel_size // 2

    mu_x = F.conv2d(prediction, kernel, padding=padding, groups=prediction.shape[1])
    mu_y = F.conv2d(target, kernel, padding=padding, groups=target.shape[1])

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(prediction ** 2, kernel, padding=padding, groups=prediction.shape[1]) - mu_x_sq
    sigma_y_sq = F.conv2d(target ** 2, kernel, padding=padding, groups=target.shape[1]) - mu_y_sq
    sigma_xy = F.conv2d(prediction * target, kernel, padding=padding, groups=prediction.shape[1]) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / (denominator + 1e-10)

    return float(ssim_map.mean().item())


def _gaussian_kernel(kernel_size: int, sigma: float, channels: int) -> torch.Tensor:
    """Build a per-channel Gaussian convolution kernel."""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    gauss_1d = torch.exp(-(x**2) / (2.0 * sigma**2))
    gauss_1d = gauss_1d / gauss_1d.sum()
    gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)  # (k, k)
    kernel = gauss_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
    kernel = kernel.expand(channels, 1, kernel_size, kernel_size).contiguous()
    return kernel
