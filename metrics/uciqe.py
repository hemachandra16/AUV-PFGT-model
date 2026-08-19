"""UCIQE (Underwater Color Image Quality Evaluation) metric.

Implements the UCIQE formulation from:
Yang & Sowmya (2015). "An underwater color image quality evaluation metric."
IEEE Transactions on Image Processing.

UCIQE = c1 * sigma_c + c2 * con_l + c3 * mu_s
"""
from __future__ import annotations

import cv2
import numpy as np
import torch


# Default weighting coefficients from the original paper
_C1 = 0.4680
_C2 = 0.2745
_C3 = 0.2576


def compute_uciqe(
    image: torch.Tensor | np.ndarray,
    c1: float = _C1,
    c2: float = _C2,
    c3: float = _C3,
) -> float:
    """Compute UCIQE for an image or batch of images.

    Args:
        image: Image tensor of shape (B, C, H, W), (C, H, W), or numpy (H, W, C).
               Expected in [0, 1] range.
        c1, c2, c3: sigma_c, con_l, mu_s weighting coefficients.

    Returns:
        Average UCIQE score over the batch.
    """
    imgs = _to_numpy_batch(image)  # (N, H, W, 3) float64 in [0, 1]
    scores = [_uciqe_single(img, c1, c2, c3) for img in imgs]
    return float(np.mean(scores))


def _uciqe_single(img: np.ndarray, c1: float, c2: float, c3: float) -> float:
    """Compute UCIQE for a single HxWx3 float image in [0, 1].

    Two corrections relative to the previous implementation, both of which made the
    metric unusable:

    1. OpenCV's 8-bit LAB stores ``a`` and ``b`` offset by +128, so a neutral grey pixel
       has (a, b) = (128, 128) rather than (0, 0). Computing ``sqrt(a^2 + b^2)`` on the
       raw channels therefore reported a colourless pixel as maximally chromatic: on a
       real UIEB reference image the mean chroma came out as 166.1 instead of 21.5, a
       7.7x inflation that swamped the sigma_c term.

    2. Saturation was computed as ``chroma / (L + 1e-10)``. OpenCV's L reaches exactly 0
       on black pixels, so any image containing pure black produced saturation values
       around 1.8e12. That is what drove whole-dataset UCIQE averages into the millions
       (a fully black frame scored 4.66e11). Saturation now uses the bounded form
       ``chroma / sqrt(chroma^2 + L^2)``, which lies in [0, 1] by construction.

    Channels are normalised to comparable [0, 1] / [-0.5, 0.5] ranges so the paper's
    weighting coefficients apply, giving scores in the usual ~0.2-0.7 band.
    """
    # Convert to LAB color space via uint8 BGR (OpenCV convention)
    img_uint8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float64)

    # OpenCV 8-bit LAB: L in [0, 255] encoding [0, 100]; a, b in [0, 255] encoding
    # [-128, 127]. Normalise L to [0, 1] and re-centre a, b to [-0.5, 0.5].
    L = lab[:, :, 0] / 255.0
    a = (lab[:, :, 1] - 128.0) / 255.0
    b = (lab[:, :, 2] - 128.0) / 255.0

    # Chroma = sqrt(a^2 + b^2)
    chroma = np.sqrt(a**2 + b**2)

    # sigma_c: standard deviation of chroma
    sigma_c = float(np.std(chroma))

    # con_l: contrast of luminance using the 1st and 99th percentiles
    l_flat = L.flatten()
    top = np.percentile(l_flat, 99)
    bot = np.percentile(l_flat, 1)
    con_l = float((top - bot) / (top + bot + 1e-10))

    # mu_s: mean saturation, in the bounded chroma / sqrt(chroma^2 + L^2) form so that
    # black pixels (L = 0) cannot produce a division blow-up.
    saturation = chroma / np.sqrt(chroma**2 + L**2 + 1e-12)
    mu_s = float(np.mean(saturation))

    return c1 * sigma_c + c2 * con_l + c3 * mu_s


def _to_numpy_batch(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert any image representation to a (N, H, W, 3) float64 array in [0,1]."""
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().float().numpy()

    image = np.asarray(image, dtype=np.float64)

    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
        image = image[np.newaxis]
    elif image.ndim == 3:
        if image.shape[0] == 3 and image.shape[2] != 3:
            image = image.transpose(1, 2, 0)
        image = image[np.newaxis]
    elif image.ndim == 4:
        if image.shape[1] == 3 and image.shape[3] != 3:
            image = image.transpose(0, 2, 3, 1)

    return image
