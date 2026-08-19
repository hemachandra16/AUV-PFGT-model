from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

try:
    from pytorch_wavelets import DWTForward, DWTInverse
except ImportError as exc:  # pragma: no cover - import-time dependency check
    raise ImportError(
        "pytorch_wavelets is required for wavelet decomposition. "
        "Install it with 'pip install pytorch-wavelets'."
    ) from exc


class WaveletTransform(nn.Module):
    """Single-level Haar wavelet transform for 4D image tensors.

    The module expects input tensors of shape (B, C, H, W) where H and W are
    divisible by 2. It performs one level of discrete wavelet decomposition
    and reconstruction using the Haar basis.
    """

    def __init__(self, wavelet: str = "haar", level: int = 1) -> None:
        super().__init__()

        if wavelet != "haar":
            raise ValueError(f"Only the Haar wavelet is supported, got '{wavelet}'.")
        if level != 1:
            raise ValueError(f"Only a single decomposition level is supported, got J={level}.")

        self.wavelet = wavelet
        self.level = level
        self.dwt = DWTForward(J=self.level, wave=self.wavelet, mode="symmetric")
        self.iwt = DWTInverse(wave=self.wavelet, mode="symmetric")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply a single-level DWT to an image tensor.

        Args:
            x: Input image tensor of shape (B, C, H, W).

        Returns:
            A tuple (LL, LH, HL, HH) containing the four wavelet sub-bands.

        Notes:
            - LL represents the low-frequency approximation band. It captures
              global illumination, smooth structures, and color information.
            - LH represents the horizontal detail band. It captures horizontal
              edges and fine-scale variations along the horizontal direction.
            - HL represents the vertical detail band. It captures vertical edges
              and fine-scale variations along the vertical direction.
            - HH represents the diagonal detail band. It captures diagonal
              textures and high-frequency corner-like structures.
        """
        self._validate_input(x)

        with torch.amp.autocast('cuda', enabled=False):
            x_fp32 = x.float()
            ll, coeffs = self.dwt(x_fp32)
            ll = ll.to(x.dtype)
            if isinstance(coeffs, list):
                coeffs = [c.to(x.dtype) if isinstance(c, torch.Tensor) else c for c in coeffs]
            elif isinstance(coeffs, torch.Tensor):
                coeffs = coeffs.to(x.dtype)
        if isinstance(coeffs, list) and len(coeffs) == 1:
            coeffs = coeffs[0]

        if isinstance(coeffs, torch.Tensor):
            if coeffs.dim() != 5:
                raise ValueError(
                    "Unexpected wavelet coefficient structure returned by pytorch_wavelets. "
                    f"Received tensor with shape {tuple(coeffs.shape)}."
                )
            lh = coeffs[:, :, 0, :, :]
            hl = coeffs[:, :, 1, :, :]
            hh = coeffs[:, :, 2, :, :]
            return ll, lh, hl, hh

        if not isinstance(coeffs, (list, tuple)) or len(coeffs) != 3:
            raise ValueError(
                "Unexpected wavelet coefficient structure returned by pytorch_wavelets. "
                f"Received {type(coeffs).__name__}."
            )

        lh, hl, hh = coeffs
        return ll, lh, hl, hh

    def inverse(
        self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct an image tensor from its four wavelet sub-bands."""
        self._validate_bands(ll, lh, hl, hh)
        device_type = ll.device.type if ll.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            ll_fp32 = ll.float()
            lh_fp32 = lh.float()
            hl_fp32 = hl.float()
            hh_fp32 = hh.float()
            highs_fp32 = torch.stack([lh_fp32, hl_fp32, hh_fp32], dim=2)
            out = self.iwt((ll_fp32, [highs_fp32]))
            return out.to(ll.dtype)

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, got {type(x).__name__}.")
        if x.dim() != 4:
            raise ValueError(f"Expected input tensor of shape (B, C, H, W), got {tuple(x.shape)}.")
        if x.shape[-2] % 2 != 0 or x.shape[-1] % 2 != 0:
            raise ValueError(
                "Wavelet decomposition requires even spatial dimensions. "
                f"Received H={x.shape[-2]} and W={x.shape[-1]}."
            )

    def _validate_bands(
        self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor
    ) -> None:
        for name, band in (("LL", ll), ("LH", lh), ("HL", hl), ("HH", hh)):
            if not isinstance(band, torch.Tensor):
                raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(band).__name__}.")
            if band.dim() != 4:
                raise ValueError(f"Expected {name} to have shape (B, C, H, W), got {tuple(band.shape)}.")

        if not (ll.shape == lh.shape == hl.shape == hh.shape):
            raise ValueError(
                "All wavelet sub-bands must have matching shapes. "
                f"Received LL={tuple(ll.shape)}, LH={tuple(lh.shape)}, "
                f"HL={tuple(hl.shape)}, HH={tuple(hh.shape)}."
            )
