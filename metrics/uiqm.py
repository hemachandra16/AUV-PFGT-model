"""UIQM (Underwater Image Quality Measure) metric.

Implements the UIQM formulation from:
Panetta et al. (2016). "Human-visual-system-inspired underwater image quality measures."
IEEE Journal of Oceanic Engineering.

UIQM = c1 * UICM + c2 * UISM + c3 * UIConM
"""
from __future__ import annotations

import numpy as np
import torch


# Default weighting coefficients from the original paper
_C1 = 0.0282
_C2 = 0.2953
_C3 = 3.5753


def compute_uiqm(
    image: torch.Tensor | np.ndarray,
    c1: float = _C1,
    c2: float = _C2,
    c3: float = _C3,
) -> float:
    """Compute UIQM for an image or batch of images.

    Args:
        image: Image tensor of shape (B, C, H, W), (C, H, W), or numpy (H, W, C).
               Expected in [0, 1] range.
        c1, c2, c3: UICM, UISM, UIConM weighting coefficients.

    Returns:
        Average UIQM score over the batch.
    """
    imgs = _to_numpy_batch(image)  # (N, H, W, 3)
    scores = [_uiqm_single(img, c1, c2, c3) for img in imgs]
    return float(np.mean(scores))


def _uiqm_single(img: np.ndarray, c1: float, c2: float, c3: float) -> float:
    """Compute UIQM for a single HxWx3 float image in [0, 1]."""
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    uicm = _compute_uicm(img_uint8)
    uism = _compute_uism(img_uint8)
    uiconm = _compute_uiconm(img_uint8)
    return c1 * uicm + c2 * uism + c3 * uiconm


def _compute_uicm(img: np.ndarray) -> float:
    """Underwater Image Colorfulness Measure (UICM)."""
    R = img[:, :, 0].astype(np.float64)
    G = img[:, :, 1].astype(np.float64)
    B = img[:, :, 2].astype(np.float64)

    RG = R - G
    YB = 0.5 * (R + G) - B

    mu_rg = np.mean(RG)
    mu_yb = np.mean(YB)
    sigma_rg = np.std(RG)
    sigma_yb = np.std(YB)

    sigma_rgyb = np.sqrt(sigma_rg**2 + sigma_yb**2)
    mu_rgyb = np.sqrt(mu_rg**2 + mu_yb**2)

    uicm = -0.0268 * sigma_rgyb + 0.1586 * mu_rgyb
    return float(uicm)


def _compute_uism(img: np.ndarray) -> float:
    """Underwater Image Sharpness Measure (UISM).

    Uses Sobel edge detection on each channel.
    """
    from scipy.ndimage import sobel as scipy_sobel  # lightweight optional

    total = 0.0
    weights = [0.299, 0.587, 0.114]  # luminance weights (R, G, B)
    for ch, w in enumerate(weights):
        channel = img[:, :, ch].astype(np.float64)
        sx = scipy_sobel(channel, axis=1)
        sy = scipy_sobel(channel, axis=0)
        edge_mag = np.hypot(sx, sy)
        total += w * _eme(edge_mag)
    return float(total)


def _compute_uiconm(img: np.ndarray) -> float:
    """Underwater Image Contrast Measure (UIConM)."""
    gray = (
        0.299 * img[:, :, 0].astype(np.float64)
        + 0.587 * img[:, :, 1].astype(np.float64)
        + 0.114 * img[:, :, 2].astype(np.float64)
    )
    return float(_logamee(gray))


def _eme(img: np.ndarray, window_size: int = 8) -> float:
    """Measure of Enhancement (EME) over a block grid."""
    h, w = img.shape
    score = 0.0
    count = 0
    for r in range(0, h - window_size, window_size):
        for c in range(0, w - window_size, window_size):
            block = img[r : r + window_size, c : c + window_size]
            mn = block.min()
            mx = block.max()
            if mn > 1e-6:
                score += 20.0 * np.log10(mx / (mn + 1e-10))
            count += 1
    return score / max(count, 1)


def _logamee(img: np.ndarray, window_size: int = 8) -> float:
    """Log-AME measure for contrast."""
    h, w = img.shape
    score = 0.0
    count = 0
    for r in range(0, h - window_size, window_size):
        for c in range(0, w - window_size, window_size):
            block = img[r : r + window_size, c : c + window_size]
            mn = block.min()
            mx = block.max()
            if mn > 1e-6 and mx > 1e-6:
                score += np.log10(mx / (mn + 1e-10)) * np.log10(mx + 1e-10)
            count += 1
    return score / max(count, 1)


def _to_numpy_batch(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert any image representation to a (N, H, W, 3) float64 array in [0,1]."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().float().numpy()

    image = np.asarray(image, dtype=np.float64)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
        image = image[np.newaxis]  # (1, H, W, 3)
    elif image.ndim == 3:
        if image.shape[0] == 3:  # (C, H, W) torch convention
            image = image.transpose(1, 2, 0)
        image = image[np.newaxis]  # (1, H, W, 3)
    elif image.ndim == 4:
        # Could be (B, C, H, W) torch or (B, H, W, C) numpy
        if image.shape[1] == 3 and image.shape[3] != 3:
            image = image.transpose(0, 2, 3, 1)  # (B, H, W, C)

    return image
