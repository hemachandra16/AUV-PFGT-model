# Project Specification

## Project Name

Physics-Aware Frequency-Guided Transformer for Underwater Image Enhancement

---

## Objective

Develop a deep learning model that enhances degraded underwater images by combining:

- Physics Prior Encoder
- Wavelet Decomposition
- Dual-Frequency Transformer
- Physics-Guided Attention
- Cross-Frequency Fusion
- Inverse Wavelet Reconstruction
- Refinement Network

The objective is to restore underwater images while preserving colors, textures, and fine details.

---

## Programming Framework

Python 3

PyTorch

---

## Dataset

Dataset Name:
UIEB (Underwater Image Enhancement Benchmark)

Folder Structure:

datasets/
└── UIEB/
    ├── raw-890/
    └── reference-890/

Input:
Raw underwater images

Target:
Reference images

Images are paired using identical filenames.

---

## Image Settings

Image Size:
256 × 256

Channels:
3 (RGB)

Image Format:
PNG

Tensor Format:
(B, C, H, W)

---

## Hardware

Train using CUDA if available.

Otherwise use CPU.

---

## Coding Standards

- Modular code
- One module per file
- Object-oriented design
- Well documented
- Clear variable names
- Type hints where appropriate
- Compatible with PyTorch 2.x

---

## Project Modules

dataset.py

wavelet.py

physics_encoder.py

low_frequency_transformer.py

high_frequency_transformer.py

physics_guided_attention.py

fusion.py

model.py

losses.py

train.py

evaluate.py

---

## Initial Milestone

The first goal is NOT training.

The first goal is verifying that paired UIEB images load correctly through a PyTorch Dataset and DataLoader.

Only after successful verification should implementation continue.

---

## Research Pipeline

Input Image
↓

Physics Prior Encoder
↓

Wavelet Decomposition
↓

Frequency Encoder
↓

Low Frequency Transformer

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

## Current Status

Dataset downloaded.

Project structure created.

Python environment configured.

Ready to implement modules one by one.