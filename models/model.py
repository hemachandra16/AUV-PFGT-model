from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fusion import FeatureFusion
from models.inverse_wavelet import InverseWaveletReconstruction
from models.physics_encoder import PhysicsPriorEncoder
from models.refinement import ImageRefinementHead
from models.transformer_block import TransformerBlock
from models.wavelet import WaveletTransform


class PFGTUIEModel(nn.Module):
    """Complete Physics-aware Frequency-Guided Transformer for underwater image enhancement.

    The model follows the architecture specified in the project documentation:
    1. Encode the input image into a physics feature map.
    2. Apply a single-level Haar wavelet transform.
    3. Process the low-frequency branch and concatenated high-frequency branch
       with transformer blocks.
    4. Fuse low-frequency, high-frequency, and physics features.
    5. Reconstruct the full-resolution feature map via inverse wavelet transform.
    6. Refine the reconstructed features into an enhanced RGB image.
    """

    def __init__(self, embed_dim: int = 128, num_heads: int = 1) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive, got {embed_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}.")

        self.physics_encoder = PhysicsPriorEncoder(in_channels=3, hidden_channels=64)
        self.wavelet = WaveletTransform(wavelet="haar", level=1)

        # Lightweight projections to match the expected input dimensions of the
        # transformer and fusion modules while preserving the intended frequency
        # structure of the wavelet bands.
        self.low_projection = nn.Conv2d(3, 128, kernel_size=1, bias=False)
        self.high_projection = nn.Conv2d(9, 384, kernel_size=1, bias=False)
        self.high_fusion_projection = nn.Conv2d(384, 128, kernel_size=1, bias=False)
        self.physics_fusion_projection = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.reconstruction_projection = nn.Conv2d(256, 128, kernel_size=1, bias=False)

        self.low_projection_norm = nn.InstanceNorm2d(128)
        self.high_projection_norm = nn.InstanceNorm2d(384)   # applied once at projection time
        self.high_feature_norm = nn.InstanceNorm2d(384)      # applied once post-transformer
        self.high_fusion_norm = nn.InstanceNorm2d(128)
        self.physics_fusion_norm = nn.InstanceNorm2d(128)
        self.reconstruction_norm = nn.InstanceNorm2d(128)

        self.low_token_norm = nn.LayerNorm(embed_dim)
        self.high_token_norm = nn.LayerNorm(embed_dim * 3)
        self.fusion_norm = nn.InstanceNorm2d(128)

        # The low-frequency branch uses a transformer block over the LL band.
        self.low_freq_transformer = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads)

        # The high-frequency branch receives the concatenated LH/HL/HH bands.
        self.high_freq_transformer = TransformerBlock(embed_dim=embed_dim * 3, num_heads=num_heads)

        self.fusion = FeatureFusion(channels=128)
        self.inverse_wavelet = InverseWaveletReconstruction(wavelet="haar")
        self.refinement_head = ImageRefinementHead(in_channels=128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the full PFGT-UIE pipeline on an RGB image.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Enhanced RGB image of shape (B, 3, H, W).
        """
        self._validate_input(x)

        # Physics prior features are extracted directly from the RGB input and
        # used as guidance for the attention modules. They do not replace the
        # learned wavelet branch features.
        physics_features = self.physics_encoder(x)  # (B, 64, H, W)

        # Single-level Haar wavelet decomposition of the input RGB image.
        ll, lh, hl, hh = self.wavelet(x)  # each: (B, 3, H/2, W/2)

        # Project the wavelet sub-bands to the embedding sizes expected by the
        # transformer blocks before turning them into tokens.
        ll_proj = self.low_projection_norm(self.low_projection(ll))
        high_proj = self.high_projection_norm(self.high_projection(torch.cat([lh, hl, hh], dim=1)))

        # Low-frequency branch: process LL with a transformer block.
        low_tokens, low_token_shape = self._reshape_to_tokens(ll_proj, embed_dim=128)  # (B, N, 128)
        low_tokens = self.low_freq_transformer(low_tokens, physics_features)
        low_tokens = self.low_token_norm(low_tokens)
        low_features = self._reshape_from_tokens(
            low_tokens,
            spatial_shape=low_token_shape,
            output_size=ll.shape[-2:],
        )
        low_features = self.fusion_norm(low_features + ll_proj)

        # High-frequency branch: concatenate LH, HL, and HH along the channel
        # dimension and project them with a learnable 1x1 convolution before
        # passing them through the transformer. This preserves directional
        # information while still learning a compact high-frequency embedding.
        high_tokens, high_token_shape = self._reshape_to_tokens(high_proj, embed_dim=384)
        high_tokens = self.high_freq_transformer(high_tokens, physics_features)
        high_tokens = self.high_token_norm(high_tokens)
        high_features = self._reshape_from_tokens(
            high_tokens,
            spatial_shape=high_token_shape,
            output_size=lh.shape[-2:],
        )
        high_features = self.high_feature_norm(high_features + high_proj)  # separate norm — not the projection norm

        # Split the processed high-frequency tensor back into LH/HL/HH while
        # preserving the learned directional bands.
        high_channels = high_features.shape[1] // 3
        lh_out = high_features[:, :high_channels, :, :]
        hl_out = high_features[:, high_channels : 2 * high_channels, :, :]
        hh_out = high_features[:, 2 * high_channels :, :, :]

        # Fuse low-frequency, high-frequency, and physics-aware features.
        physics_fused = self.physics_fusion_norm(
            self.physics_fusion_projection(physics_features)
        )
        physics_fused = F.interpolate(
            physics_fused,
            size=low_features.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        high_fused = self.high_fusion_norm(
            self.high_fusion_projection(torch.cat([lh_out, hl_out, hh_out], dim=1))
        )
        fused_features = self.fusion(low_features, high_fused, physics_fused)
        fused_features = self.fusion_norm(fused_features)

        # Reconstruct the full-resolution feature map with inverse wavelet transform.
        reconstructed = self.inverse_wavelet(
            ll=low_features,
            lh=lh_out,
            hl=hl_out,
            hh=hh_out,
        )

        # Inject the fused cross-frequency representation into the final
        # reconstruction path through explicit concatenation and a learned
        # projection. This preserves the original inverse-wavelet reconstruction
        # while avoiding unstable cross-scale addition between different feature
        # origins.
        fused_for_reconstruction = F.interpolate(
            fused_features,
            size=reconstructed.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        refinement_input = torch.cat([reconstructed, fused_for_reconstruction], dim=1)
        refinement_input = self.reconstruction_norm(
            self.reconstruction_projection(refinement_input)
        )

        # Refine the reconstructed features into the final RGB image.
        enhanced = self.refinement_head(refinement_input)
        return torch.clamp(enhanced, min=0.0, max=1.0)

    def _reshape_to_tokens(self, x: torch.Tensor, embed_dim: int) -> tuple[torch.Tensor, tuple[int, int]]:
        """Convert a feature map to a token sequence of shape (B, N, C)."""
        if x.dim() != 4:
            raise ValueError(f"Expected feature map with shape (B, C, H, W), got {tuple(x.shape)}.")
        batch_size, channels, height, width = x.shape
        if channels != embed_dim:
            raise ValueError(
                f"Expected {embed_dim} channels for tokenization, got {channels}."
            )

        # Preserve spatial detail with high-resolution token grid (up to 64x64)
        target_h = min(height, 64)
        target_w = min(width, 64)
        if (target_h, target_w) != (height, width):
            pooled = F.adaptive_avg_pool2d(x, (target_h, target_w))
            tokens = pooled.flatten(2).transpose(1, 2)
        else:
            tokens = x.flatten(2).transpose(1, 2)
        return tokens, (target_h, target_w)

    def _reshape_from_tokens(
        self,
        x: torch.Tensor,
        spatial_shape: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        """Convert a token sequence back to a feature map."""
        if x.dim() != 3:
            raise ValueError(f"Expected token tensor of shape (B, N, C), got {tuple(x.shape)}.")
        batch_size, seq_len, channels = x.shape
        height, width = spatial_shape
        expected_seq_len = height * width
        if seq_len != expected_seq_len:
            raise ValueError(
                f"Expected {expected_seq_len} tokens for spatial shape {spatial_shape}, got {seq_len}."
            )
        feature_map = x.transpose(1, 2).contiguous().view(batch_size, channels, height, width)
        if (height, width) != output_size:
            feature_map = F.interpolate(feature_map, size=output_size, mode="bilinear", align_corners=False)
        return feature_map

    def _validate_input(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected x to be a torch.Tensor, got {type(x).__name__}.")
        if x.dim() != 4:
            raise ValueError(f"Expected input tensor of shape (B, 3, H, W), got {tuple(x.shape)}.")
        if x.shape[1] != 3:
            raise ValueError(f"Expected 3 input channels, got {x.shape[1]}.")
