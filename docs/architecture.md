# Architecture Specification

## Project Codename

**PFGT-UIE** (Physics-aware Frequency-Guided Transformer for Underwater Image Enhancement)

---

# Objective

The objective of this project is to enhance degraded underwater images by integrating underwater imaging physics with frequency-domain feature learning and transformer-based attention.

Unlike conventional underwater image enhancement networks, the proposed architecture does not use physics as a preprocessing step. Instead, it incorporates physics-derived degradation features directly into the transformer attention mechanism.

---

# Overall Pipeline

Input RGB Image
↓

Physics Prior Encoder
↓

Physics Feature Map
↓

Discrete Wavelet Transform (DWT)
↓

LL | LH | HL | HH

↓

Dual Frequency Processing

• Low Frequency Transformer (LL)

• High Frequency Transformer (LH, HL, HH)

↓

Physics-Guided Attention

↓

Cross-Frequency Feature Fusion

↓

Inverse Wavelet Transform (IWT)

↓

Refinement Network

↓

Enhanced RGB Image

---

# Module 1 — Physics Prior Encoder

Purpose:

Estimate latent underwater degradation features from the input image.

The encoder learns representations related to:

- Light attenuation
- Forward scattering
- Back scattering
- Color degradation

Output:

Physics Feature Map

Target Channels:

64

The encoder remains lightweight to reduce computational complexity.

---

# Module 2 — Wavelet Decomposition

The RGB image is decomposed using a single-level Discrete Wavelet Transform.

Outputs:

LL

Low-frequency information

Contains:

- Global illumination
- Color information
- Smooth structures

LH

Horizontal edges

HL

Vertical edges

HH

Diagonal textures

The decomposition allows different enhancement strategies for color and detail.

---

# Module 3 — Low Frequency Transformer

Input:

LL

Learns:

- Color correction
- Illumination recovery
- Haze reduction

Physics guidance is injected into every transformer block.

---

# Module 4 — High Frequency Transformer

Input:

LH

HL

HH

Learns:

- Texture restoration
- Edge enhancement
- Fine detail recovery

This branch focuses on recovering structures lost due to underwater scattering.

---

# Module 5 — Physics-Guided Attention

This is the core novelty of the project.

Instead of computing attention solely from learned features, the transformer receives additional guidance from the Physics Prior Encoder.

Physics features are projected into an attention bias that modifies the attention scores before softmax.

This enables the network to focus on regions with severe degradation.

---

# Module 6 — Cross-Frequency Fusion

Inputs:

- Low-frequency features
- High-frequency features
- Physics features

Output:

Unified feature representation

The fusion module combines complementary information before reconstruction.

---

# Module 7 — Reconstruction

The reconstructed frequency bands are passed through an Inverse Wavelet Transform.

A lightweight refinement network removes remaining artifacts.

The final output is an enhanced RGB image.

---

# Design Principles

The architecture follows these principles:

- Modular implementation
- Lightweight components
- Physics-aware attention
- Frequency-aware processing
- End-to-end trainable
- Real-time deployment potential for AUV vision systems