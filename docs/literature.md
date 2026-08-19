# Literature Review

# Project

Physics-Aware Frequency-Guided Transformer for Underwater Image Enhancement (PFGT-UIE)

---

# Objective

This document summarizes the key literature reviewed during the development of the proposed architecture and identifies the research gap addressed by the project.

---

# Paper Category 1: Physics-Based Underwater Image Enhancement

These methods use underwater image formation models, transmission estimation, or physical priors to restore underwater images.

Strengths:
- Physically interpretable
- Good color correction
- Explainable restoration

Limitations:
- Often rely on inaccurate transmission estimation
- Sensitive to water conditions
- Limited representation learning

Research Gap:
Physics is usually used as preprocessing or explicit restoration rather than as a guidance signal inside deep neural networks.

---

# Paper Category 2: CNN-Based Enhancement

These methods use convolutional neural networks to directly map degraded underwater images to enhanced outputs.

Strengths:
- Fast inference
- End-to-end learning
- Good local feature extraction

Limitations:
- Limited long-range dependency modeling
- Difficulty recovering complex degradation

Research Gap:
CNNs struggle to model global contextual relationships in severely degraded underwater scenes.

---

# Paper Category 3: Transformer-Based Enhancement

Recent methods employ Vision Transformers to improve global feature representation.

Strengths:
- Strong global context modeling
- Better long-range feature interaction

Limitations:
- Attention relies only on learned image features
- No explicit incorporation of underwater imaging physics

Research Gap:
Transformer attention remains purely data-driven.

---

# Paper Category 4: Frequency-Based Enhancement

Several works use wavelet transforms or frequency decomposition to separately process low-frequency and high-frequency information.

Strengths:
- Better preservation of textures
- Improved edge recovery
- Reduced information loss

Limitations:
- Frequency information is not guided by physical degradation.

Research Gap:
Frequency decomposition alone cannot distinguish physically degraded regions.

---

# Identified Research Gap

Current underwater enhancement methods generally fall into one of four categories:

- Physics-based restoration
- CNN-based enhancement
- Transformer-based enhancement
- Frequency-domain enhancement

However, existing approaches rarely integrate all of the following in a unified framework:

- Physics-aware feature extraction
- Wavelet frequency decomposition
- Physics-guided transformer attention
- Cross-frequency feature fusion

---

# Proposed Contribution

The proposed PFGT-UIE framework introduces:

1. A lightweight Physics Prior Encoder that learns degradation-aware features.

2. Frequency-domain image processing using Discrete Wavelet Transform.

3. Dual transformer branches specialized for low-frequency and high-frequency information.

4. Physics-Guided Attention where physics features modify transformer attention scores.

5. Cross-frequency fusion before image reconstruction.

---

# Expected Contribution

The proposed architecture aims to:

- Improve underwater color restoration.
- Recover fine textures.
- Preserve structural details.
- Increase PSNR and SSIM.
- Improve underwater image quality metrics such as UIQM and UCIQE.

---

# Summary

The novelty of this work lies in treating underwater physics as an attention guidance mechanism rather than a preprocessing operation. This allows the transformer to focus computational resources on the most degraded regions while simultaneously exploiting frequency-domain representations for enhanced restoration.