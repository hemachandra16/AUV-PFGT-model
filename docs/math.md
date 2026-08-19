# Mathematical Formulation

# Project

Physics-Aware Frequency-Guided Transformer for Underwater Image Enhancement (PFGT-UIE)

---

# 1. Underwater Image Formation

The observed underwater image is modeled as

I(x) = J(x) · t(x) + B(x) · (1 − t(x))

where

I(x) : observed underwater image

J(x) : latent clean image

t(x) : transmission map

B(x) : background light

The objective of the network is to estimate an enhanced approximation of J(x).

---

# 2. Physics Prior Encoder

The Physics Prior Encoder is represented as

P = E(I)

where

E(.) denotes the lightweight physics encoder

Input:

RGB Image

Output:

Physics Feature Map

P ∈ R^(64×H×W)

The encoder learns degradation-related features including

• attenuation

• scattering

• color distortion

without explicitly estimating physical parameters.

---

# 3. Wavelet Decomposition

The input image is decomposed using a single-level Discrete Wavelet Transform

DWT(I)

↓

LL

LH

HL

HH

where

LL represents low-frequency information

LH represents horizontal details

HL represents vertical details

HH represents diagonal textures

---

# 4. Frequency Branches

Low-frequency branch

F_low = T_low(LL)

High-frequency branch

F_high = T_high(LH, HL, HH)

where

T_low

and

T_high

represent transformer feature extractors.

---

# 5. Physics-Guided Attention

Standard attention

Attention(Q,K,V)

=

Softmax(QKᵀ / √d)

V

The proposed method modifies attention using physics guidance

Attention

=

Softmax(

QKᵀ / √d

+

λP

)

V

where

P

is the projected physics feature map

λ

controls the influence of physics guidance.

This is the core novelty of the proposed method.

---

# 6. Cross-Frequency Fusion

The fused feature representation is

F

=

Fusion(

F_low,

F_high,

P

)

where Fusion represents the feature integration module.

---

# 7. Reconstruction

The enhanced image is reconstructed by

Î

=

Refine(

IWT(F)

)

where

IWT

is the Inverse Wavelet Transform

Refine

is the refinement CNN.

---

# 8. Loss Function

The total training loss is

L_total

=

λ1 L1

+

λ2 L_perceptual

+

λ3 L_SSIM

+

λ4 L_frequency

where

L1

ensures pixel accuracy

Perceptual Loss preserves semantic appearance

SSIM preserves structural similarity

Frequency Loss encourages wavelet consistency.

---

# Optimization

Optimizer

AdamW

Learning Rate

1 × 10⁻⁴

Batch Size

4–8

Epochs

150

Mixed Precision

Enabled

Gradient Clipping

1.0