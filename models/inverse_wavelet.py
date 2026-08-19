from __future__ import annotations

import torch
import torch.nn as nn

try:
    from pytorch_wavelets import DWTInverse
except ImportError as exc:  # pragma: no cover - import-time dependency check
    raise ImportError(
        "pytorch_wavelets is required for inverse wavelet reconstruction. "
        "Install it with 'pip install pytorch-wavelets'."
    ) from exc


class InverseWaveletReconstruction(nn.Module):
    """Reconstruct a spatial feature map from four wavelet sub-bands.

    This module uses a single-level Haar inverse DWT to upsample and combine the
    low-frequency band (LL) with horizontal, vertical, and diagonal detail bands
    (LH, HL, HH) into a full-resolution feature tensor.
    """

    def __init__(self, wavelet: str = "haar") -> None:
        super().__init__()

        if wavelet != "haar":
            raise ValueError(f"Only the Haar wavelet is supported, got '{wavelet}'.")

        self.wavelet = wavelet
        self.iwt = DWTInverse(wave=self.wavelet, mode="symmetric")

    def forward(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct a feature map from the four wavelet sub-bands.

        Args:
            ll: Low-frequency sub-band of shape (B, C, H, W).
            lh: Horizontal detail sub-band of shape (B, C, H, W).
            hl: Vertical detail sub-band of shape (B, C, H, W).
            hh: Diagonal detail sub-band of shape (B, C, H, W).

        Returns:
            Reconstructed tensor of shape (B, C, 2H, 2W).
        """
        self._validate_inputs(ll, lh, hl, hh)

        # Disable autocast during IDWT as pytorch-wavelets doesn't support fp16 well
        device_type = ll.device.type if ll.is_cuda else "cpu"
        with torch.amp.autocast(device_type=device_type, enabled=False):
            ll_fp32 = ll.float()
            lh_fp32 = lh.float()
            hl_fp32 = hl.float()
            hh_fp32 = hh.float()
            highs_fp32 = torch.stack([lh_fp32, hl_fp32, hh_fp32], dim=2)
            out = self.iwt((ll_fp32, [highs_fp32]))
            return out.to(ll.dtype)

    def _validate_inputs(
        self,
        ll: torch.Tensor,
        lh: torch.Tensor,
        hl: torch.Tensor,
        hh: torch.Tensor,
    ) -> None:
        for name, tensor in (("LL", ll), ("LH", lh), ("HL", hl), ("HH", hh)):
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Expected {name} to be a torch.Tensor, got {type(tensor).__name__}.")
            if tensor.dim() != 4:
                raise ValueError(f"Expected {name} to have shape (B, C, H, W), got {tuple(tensor.shape)}.")

        if not (ll.shape == lh.shape == hl.shape == hh.shape):
            raise ValueError(
                "All wavelet sub-bands must have matching shapes. "
                f"Received LL={tuple(ll.shape)}, LH={tuple(lh.shape)}, "
                f"HL={tuple(hl.shape)}, HH={tuple(hh.shape)}."
            )

        if ll.shape[-2] <= 0 or ll.shape[-1] <= 0:
            raise ValueError("Wavelet sub-band spatial dimensions must be positive.")
