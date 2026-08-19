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

### Implementation notes

Queries, keys and values are **distinct learned linear projections** of the input tokens,
split across `model.num_heads` heads, followed by an output projection. This matters more
than it may appear: if Q, K and V are the same tensor, `Softmax(QK'/sqrt(d))V` reduces to a
row-stochastic mixing of the input tokens, so its output is always a convex combination of
values already present. Such an operator can smooth or re-weight, but it cannot apply the
global colour shift that underwater colour correction fundamentally requires.
`tools/verify_attention.py` demonstrates this directly.

The physics bias `P` is the physics feature map projected by a 1x1 convolution to one
channel **per attention head**, pooled onto the token grid and standardised, giving a
per-head, per-position additive bias of shape `(B, heads, 1, N)` that is added to the
logits before softmax. Because it broadcasts over the query axis, the `N x N` score matrix
is never materialised — `scaled_dot_product_attention` handles it with its memory-efficient
kernels, which is what keeps the model inside an 8 GB VRAM budget at N = 4096 tokens.

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

# Module 8 — Underwater Object Detection (pipeline extension)

Enhancement is a means, not the end: the AUV use case needs to *find* things. A detection
stage therefore runs on the enhanced frame.

```
raw frame -> PFGT-UIE enhancement -> underwater detector -> annotated detections
```

The detector is a YOLO model fine-tuned on **RUOD** (Real-world Underwater Object
Detection): 14,000 real underwater images across 10 marine classes — holothurian, echinus,
scallop, starfish, fish, corals, diver, cuttlefish, turtle, jellyfish.

It is deliberately a **separate stage rather than a head on the enhancement network**. The
two tasks have different supervision (UIEB provides paired raw/reference images with no
boxes; RUOD provides boxes with no enhancement targets), so there is no dataset on which a
joint model could be trained end-to-end. Keeping them separate also lets either component
be swapped or evaluated independently, which is what the enhancement-helps-detection
ablation needs.

Implementation: `models/object_detection.py` (`build_detector`), driven by
`infer_detection.py`; training in `tools/train_detector.py`; metrics in
`results/detection_metrics.json`.

> Historical note: an earlier revision used a COCO-pretrained Faster R-CNN whose predicted
> class names were rewritten through a hand-written dictionary (`"frisbee" -> "starfish"`,
> `"bear" -> "marine_life"`). That model had never seen an underwater image, and the
> renaming added no underwater knowledge. It is retained only as a comparison baseline
> (`--detector fasterrcnn`).

---

# Design Principles

The architecture follows these principles:

- Modular implementation
- Lightweight components
- Physics-aware attention
- Frequency-aware processing
- End-to-end trainable
- Real-time deployment potential for AUV vision systems