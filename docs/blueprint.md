# Engineering Blueprint

## Project

Physics-Aware Frequency-Guided Transformer for Underwater Image Enhancement (PFGT-UIE)

---

# Objective

Develop a modular PyTorch implementation of a physics-aware frequency-guided transformer for underwater image enhancement.

The project will be implemented incrementally, with each module verified independently before integration.

---

# Project Pipeline

Input Image
↓

Dataset Loader

↓

Physics Prior Encoder

↓

Discrete Wavelet Transform

↓

Low Frequency Transformer

+

High Frequency Transformer

↓

Physics-Guided Attention

↓

Cross-Frequency Fusion

↓

Inverse Wavelet Transform

↓

Refinement Network

↓

Enhanced Image

---

# Module Implementation Order

## Stage 1

Dataset Loader

File:

data/dataset.py

Status:

Completed

---

## Stage 2

Wavelet Processing

Files:

models/wavelet.py

Purpose:

- Implement Discrete Wavelet Transform (DWT)
- Implement Inverse Wavelet Transform (IWT)
- Verify that IWT(DWT(image)) reconstructs the original image

---

## Stage 3

Physics Prior Encoder

File:

models/physics_encoder.py

Purpose:

Extract degradation-aware feature maps from underwater images.

Output:

Physics feature tensor with 64 channels.

---

## Stage 4

Low-Frequency Transformer

File:

models/low_frequency_transformer.py

Purpose:

Restore illumination, color, and low-frequency structures.

---

## Stage 5

High-Frequency Transformer

File:

models/high_frequency_transformer.py

Purpose:

Recover edges, textures, and fine image details.

---

## Stage 6

Physics-Guided Attention

File:

models/attention.py

Purpose:

Inject physics features into transformer attention using an additive attention bias.

This module represents the core scientific contribution of the project.

---

## Stage 7

Cross-Frequency Fusion

File:

models/fusion.py

Purpose:

Fuse low-frequency, high-frequency, and physics-aware features.

---

## Stage 8

Reconstruction

Files:

models/reconstruction.py

models/refinement.py

Purpose:

Perform inverse wavelet reconstruction and remove remaining artifacts.

---

## Stage 9

Complete Model

File:

models/model.py

Purpose:

Integrate all modules into a single trainable network.

---

## Stage 10

Training

File:

train.py

Purpose:

Train the complete network using paired UIEB images.

---

## Stage 11

Evaluation

File:

evaluate.py

Metrics:

- PSNR
- SSIM
- UIQM
- UCIQE

---

# Development Rules

- Build one module at a time.
- Verify every module independently.
- Do not implement future modules before the current module is tested.
- Maintain compatibility with Python 3 and PyTorch 2.x.
- Keep the implementation modular and well documented.

---

# Final Goal

Produce a reproducible, research-quality implementation suitable for publication and future deployment on underwater robotic vision systems.