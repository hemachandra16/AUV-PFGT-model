from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PFGTLoss(nn.Module):
    """Composite loss for the Physics-aware Frequency-Guided Transformer.

    The total objective combines:
    - L1 reconstruction loss for pixel-level accuracy
    - SSIM-based structural loss for perceptual consistency
    - Perceptual loss from pretrained VGG19 features for semantic similarity

    The module returns a dictionary of component losses and the summed total.
    """

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_ssim: float = 0.5,
        lambda_perceptual: float = 0.1,
    ) -> None:
        super().__init__()

        if lambda_l1 < 0:
            raise ValueError(f"lambda_l1 must be non-negative, got {lambda_l1}.")
        if lambda_ssim < 0:
            raise ValueError(f"lambda_ssim must be non-negative, got {lambda_ssim}.")
        if lambda_perceptual < 0:
            raise ValueError(f"lambda_perceptual must be non-negative, got {lambda_perceptual}.")

        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_perceptual = lambda_perceptual

        # Use a pretrained VGG19 feature extractor for perceptual loss.
        # The model is frozen by default to preserve the pretrained semantics.
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        self.vgg_features = nn.Sequential(*list(vgg19.features.children())[:16])
        for parameter in self.vgg_features.parameters():
            parameter.requires_grad = False

        self.vgg_features.eval()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute the composite loss for a prediction-target pair.

        Args:
            prediction: Predicted image tensor of shape (B, 3, H, W).
            target: Target image tensor of shape (B, 3, H, W).

        Returns:
            Dictionary containing the total loss and the individual components.
        """
        self._validate_inputs(prediction, target)

        l1_loss = self._l1_loss(prediction, target)
        ssim_loss = self._ssim_loss(prediction, target)
        perceptual_loss = self._perceptual_loss(prediction, target)

        total_loss = (
            self.lambda_l1 * l1_loss
            + self.lambda_ssim * ssim_loss
            + self.lambda_perceptual * perceptual_loss
        )

        return {
            "total_loss": total_loss,
            "l1_loss": l1_loss,
            "ssim_loss": ssim_loss,
            "perceptual_loss": perceptual_loss,
        }

    def _l1_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pixel-wise L1 reconstruction loss."""
        return F.l1_loss(prediction, target)

    def _ssim_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Structural similarity loss based on the SSIM formulation."""
        # Compute SSIM over each image independently and optimize the complement.
        ssim_value = self._ssim(prediction, target)
        return 1.0 - ssim_value

    def _perceptual_loss(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Perceptual loss computed from VGG19 feature activations.

        The feature-space loss is kept in an L1 form to avoid the very large
        magnitudes that can otherwise dominate the total objective during early
        training, especially when the VGG19 features are activated on image-sized
        tensors.
        """
        with torch.no_grad():
            target_features = self._extract_vgg_features(target)

        prediction_features = self._extract_vgg_features(prediction)
        return F.l1_loss(prediction_features, target_features)

    def _extract_vgg_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract intermediate VGG19 features for an image tensor."""
        if image.shape[1] != 3:
            raise ValueError(f"Expected 3 input channels, got {image.shape[1]}.")

        # Ensure image is float32 for VGG19 feature extraction under AMP.
        image_fp32 = image.float()
        normalized = self._normalize_to_vgg(image_fp32)
        features = self.vgg_features(normalized)
        return features

    def _normalize_to_vgg(self, image: torch.Tensor) -> torch.Tensor:
        """Normalize image tensors to the VGG19 input statistics."""
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        return (image - mean) / std

    def _ssim(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the mean SSIM over a batch of image tensors."""
        # SSIM uses a small stabilizer to avoid division by zero.
        c1 = 0.01**2
        c2 = 0.03**2

        mu_x = F.avg_pool2d(prediction, kernel_size=11, stride=1, padding=5)
        mu_y = F.avg_pool2d(target, kernel_size=11, stride=1, padding=5)

        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x_sq = F.avg_pool2d(prediction.pow(2), kernel_size=11, stride=1, padding=5) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(target.pow(2), kernel_size=11, stride=1, padding=5) - mu_y_sq
        sigma_xy = F.avg_pool2d(prediction * target, kernel_size=11, stride=1, padding=5) - mu_x_mu_y

        numerator = (2.0 * mu_x_mu_y + c1) * (2.0 * sigma_xy + c2)
        denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
        ssim_map = numerator / (denominator + 1e-8)
        return ssim_map.mean()

    def _validate_inputs(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        """Validate prediction and target tensor shapes and dtypes."""
        if not isinstance(prediction, torch.Tensor):
            raise TypeError(f"Expected prediction to be a torch.Tensor, got {type(prediction).__name__}.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"Expected target to be a torch.Tensor, got {type(target).__name__}.")

        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction and target must have the same shape, got {tuple(prediction.shape)} and {tuple(target.shape)}."
            )
        if prediction.dim() != 4:
            raise ValueError(f"Expected input tensors of shape (B, C, H, W), got {tuple(prediction.shape)}.")
        if prediction.shape[1] != 3:
            raise ValueError(f"Expected 3 input channels, got {prediction.shape[1]}.")
