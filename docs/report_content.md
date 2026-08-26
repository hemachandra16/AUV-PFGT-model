# PFGT-UIE

## Physics-Guided Frequency Transformer for Underwater Image Enhancement, with Marine Object Detection

**A seven-session engineering and research record.** Held-out enhancement: **25.364 dB PSNR / 0.9289 SSIM**. Detection: **mAP@0.5 = 0.829** across ten marine classes.

---

## 1. What this project is

Underwater photographs are physically degraded before they ever reach a sensor. Water absorbs red light within a few metres of depth while blue and green persist, and suspended particles scatter light back into the lens as a veiling haze. The result is the familiar blue-green cast, washed-out contrast, and lost detail. For a diver this is cosmetic; for an autonomous underwater vehicle trying to recognise what it is looking at, it is a perception problem.

PFGT-UIE is a two-part system for that setting. The first part is an **enhancement network** that reverses the colour and contrast degradation, built around a physics-guided transformer operating on wavelet frequency sub-bands. The second is a **detector** that finds and classifies marine objects — fish, coral, divers, seafloor invertebrates — in real time on the kind of hardware an AUV can carry.

The two parts turned out to interact in a way that is worth stating at the top, because it is counter-intuitive and it changed the system's design: **running the enhancer before the detector makes detection worse, not better.** The pipeline therefore detects on raw frames and uses enhancement for the human-facing view. Section 6 covers why.

---

## 2. The enhancement architecture

The network takes a degraded RGB image and produces an enhanced one at the same resolution. It has 2,308,723 parameters — small enough to be plausible on embedded hardware.

```
input RGB
   ├─────────────────────────────────► Physics Prior Encoder ──► P (64ch) + global context
   │                                                              │
   └─► Haar DWT ──┬─► LL  ──► Low-Frequency Transformer  ◄────────┤
                  │                (physics-guided attention)     │
                  └─► LH,HL,HH ─► High-Frequency Transformer ◄────┘
                                          │
                        Cross-Frequency Fusion (low, high, physics)
                                          │
                          Inverse Haar DWT reconstruction
                                          │
                             Refinement Head (GroupNorm)
                                          │
                        Global Colour Correction (per-image affine)
                                          │
                                    enhanced RGB
```

### 2.1 Physics Prior Encoder

This module exists to give the rest of the network a signal about *how* the image is degraded, not just what it contains.

**What it was originally.** A plain stack: `Conv3×3(3→64) → GELU → 2× residual block → Conv3×3(64→64)`. Its only input was the same RGB the wavelet branch already received. It was named "physics encoder" but contained no physics — it could only re-derive what the rest of the network could compute for itself. Worse, every layer was a 3×3 convolution with no global pooling anywhere, which means image-wide per-channel statistics were **not representable at any depth**. Those statistics are precisely what a colour cast is.

**What it is now.** Eight closed-form degradation priors are computed analytically and concatenated to the RGB input before the convolution stack:

| Prior | What it measures |
|---|---|
| Dark channel | Per-pixel minimum across colour channels, locally min-pooled. Underwater this is dominated by backscatter, so it tracks transmission and depth. |
| Bright channel | Local maximum. Tracks the veiling light / background illuminant. |
| Local contrast | Local standard deviation. Scattering suppresses local contrast, so this is a proxy for how much haze sits in front of a region. |
| Per-channel means (×3) | Broadcast image-wide channel means — the direct measurement of the colour cast. |
| R/G and R/B ratios (×2) | Broadcast attenuation ratios. Red is absorbed fastest with depth, blue least, so these encode an effective depth cue that is invariant to exposure. |

A global-average-pool branch was also added, producing a context vector that both modulates the feature map (squeeze-and-excitation style) and is passed downstream to the colour-correction module.

**Why this matters, measured.** Regressing those eight numbers against the per-channel gain the ground-truth reference actually applies gives R² = 0.24 / 0.49 / 0.62 for red, green and blue. Roughly a quarter to two-thirds of the required colour correction is linearly predictable from eight scalars the original module was structurally incapable of computing.

### 2.2 Haar wavelet decomposition

A single-level Haar discrete wavelet transform splits the image into four sub-bands at half resolution:

- **LL** — the low-frequency approximation. Carries global illumination, colour, and smooth structure. This is where underwater colour degradation lives.
- **LH, HL, HH** — horizontal, vertical and diagonal detail. Edges and texture.

Separating them lets the network apply different processing to "fix the colour" and "restore the texture", which are genuinely different problems. The transform is orthogonal and exactly invertible, so nothing is lost by the split itself.

### 2.3 The dual transformer branches and physics-guided attention

Each branch tokenises its sub-band into a grid and runs a pre-LayerNorm transformer block. The low-frequency branch operates on LL at embedding dimension 128; the high-frequency branch on the concatenated LH/HL/HH at dimension 384.

The attention is where the "physics-guided" claim lives:

```math
Attention(Q, K, V)  =  Softmax( QKᵀ / √d  +  λP ) V
```

`P` is the physics feature map, projected by a 1×1 convolution to one channel per attention head, pooled onto the token grid and standardised. `λ` is a learned scalar. The bias is added to the attention logits **before** the softmax, so regions the physics encoder marks as heavily degraded can attract or repel attention.

**The core bug, and why it mattered.** In the original implementation there were **no Q, K or V projections at all**. `transformer_block.py` passed the same tensor as query, key and value, and the attention module's entire learnable parameter count was 66 — a one-channel physics convolution plus one scalar.

With Q = K = V, `Softmax(QKᵀ/√d)V` is a row-stochastic matrix applied to the tokens themselves. Its output is therefore always a **convex combination of values already present in the input**. It can smooth, re-weight, or blur — but it can never move a value outside the range its own input already spans. Removing a global blue cast is exactly such a move. The module was mathematically incapable of the operation the architecture was named for.

This is demonstrated rather than asserted. `tools/verify_attention.py` asks each version to learn the simplest possible colour correction, `tokens → tokens + 3.0`:

```
OLD module learnable params :    66   (no Q/K/V projections at all)
NEW module learnable params : 66,309
input  mean : -0.0069     target mean : +2.9931
OLD output mean after fitting: -0.0069   final MSE: 9.00000   ← 9.0 = 3.0², learned NOTHING
NEW output mean after fitting: +2.9943   final MSE: 0.39800
```

The old module's error equals the square of the shift it was asked to learn, and its output mean never left the input mean: zero progress, as predicted analytically.

The fix adds real learned `q_proj`, `k_proj`, `v_proj` and a standard multi-head `out_proj`, plus a genuine head split. The physics bias was simultaneously upgraded from a rank-1 scalar outer product to a per-head, per-position projected bias of shape `(B, heads, 1, N)` — the faithful reading of "P is the projected physics feature map".

A useful side effect: because that bias broadcasts over the query axis, `scaled_dot_product_attention` never materialises the N×N score matrix. Peak VRAM at batch size 8 **fell from 7.19 GB to 5.19 GB** while parameters increased.

### 2.4 Cross-frequency fusion

The processed low-frequency, high-frequency and physics features are concatenated and projected back to a common width, refined by a residual block. Both the fusion block and the refinement head originally used `BatchNorm2d`, which at batch size 8 makes the network train a measurably different function from the one it is evaluated as — worth ~0.42 dB in the refinement head alone. Both were changed to `GroupNorm`, which is batch-independent, so train and eval compute identically.

### 2.5 Inverse wavelet reconstruction and refinement head

The inverse Haar DWT reassembles a full-resolution feature map, which a small decoder (128 → 64 → 32 → 3, GroupNorm, GELU, sigmoid) turns into the output image.

The head is deliberately small — 4.8% of parameters. That is a known imbalance: measured by decomposing the model's remaining error into wavelet bands, **97.3% of the residual error is in the LL band** (colour and illumination), yet the high-frequency transformer originally held 65% of all parameters to serve the 2.7% in the detail bands. Halving the high branch's FFN ratio freed ~590k parameters and made the model 15.4% smaller overall.

### 2.6 Global colour correction

A late module predicts a per-image, per-channel affine (gain and shift) from the physics context vector and applies it to the decoded image. It is initialised to an exact identity, so at step zero it changes nothing.

It exists because of a measurement: applying an *oracle* per-image per-channel offset — computed using the ground-truth reference — is worth **+3.20 dB**, and 43% of the model's remaining error energy is a single per-image constant. Nothing in the network could emit one: the refinement head's input is instance-normalised to exactly zero per-image per-channel mean, its receptive field is 9×9, and every convolution in it was bias-free.

**The honest outcome.** In isolation the module is expressive enough — frozen everything else and fitted only its 4,550 parameters against the oracle, it recovers **91.7%** of the headroom. But end-to-end it converged to a nearly constant tone adjustment, and a later attempt to force it with a dedicated loss term made the model *worse* (−0.382 dB). The reason was then measured: the required offset is only weakly predictable from the input at all (held-out ridge R² = 0.015 / 0.104 / 0.346), because UIEB's reference images are hand-retouched and much of their colour reflects a human editor's taste rather than a recoverable property of the photograph. Against that, only **24.4% of the apparent +3.20 dB headroom is reachable** by any input-conditioned method. Section 4 covers this.

---

## 3. The detection architecture

### 3.1 What it replaced

The original "underwater object detector" was a COCO-pretrained Faster R-CNN whose predictions were renamed through a hand-written dictionary:

```python
"frisbee": "starfish",   "kite": "stingray",   "bear": "marine_life",
"banana": "sea_cucumber", "umbrella": "jellyfish", "toothbrush": "underwater_debris", ...
```

That model had never seen an underwater image. The renaming adds no underwater knowledge — it relabels whatever everyday object the network happened to fire on, so a real starfish is only ever called one if it first looks like a frisbee. There was no fine-tuning code, no underwater dataset, and no detection metric anywhere in the repository.

Run on real underwater frames, it behaves exactly as that description predicts: on three held-out images it produced *bird*, *frisbee*, *spoon* and *toothbrush* labels, or found nothing at all.

### 3.2 What it is now

**YOLO11n fine-tuned on RUOD** (Real-world Underwater Object Detection): 14,000 real underwater photographs, 9,800 train / 4,200 held-out, across ten genuine marine classes — holothurian, echinus, scallop, starfish, fish, corals, diver, cuttlefish, turtle, jellyfish.

A lightweight single-stage detector is the right choice for this deployment context rather than a compromise. An AUV has a hard power budget, no thermal headroom for a large GPU, and needs per-frame latency low enough to act on. YOLO11n is 2.6 M parameters and 6.4 GFLOPs — comparable in size to the enhancement network — and runs single-shot with no region-proposal stage. A two-stage detector like Faster R-CNN is roughly an order of magnitude heavier for accuracy that this task does not obviously need.

### 3.3 Results

**mAP@0.5 = 0.8292 · mAP@0.5:0.95 = 0.5845 · precision 0.8385 · recall 0.7561** over 4,200 held-out images and 22,968 object instances.

| Class | AP@0.5 | AP@0.5:0.95 | Precision | Recall |
|---|---|---|---|---|
| cuttlefish | 0.965 | 0.818 | 0.939 | 0.927 |
| turtle | 0.965 | 0.826 | 0.932 | 0.924 |
| diver | 0.929 | 0.715 | 0.883 | 0.883 |
| echinus | 0.880 | 0.493 | 0.886 | 0.812 |
| starfish | 0.862 | 0.518 | 0.829 | 0.806 |
| jellyfish | 0.787 | 0.608 | 0.661 | 0.781 |
| holothurian | 0.751 | 0.444 | 0.834 | 0.641 |
| fish | 0.746 | 0.500 | 0.802 | 0.640 |
| scallop | 0.714 | 0.425 | 0.826 | 0.556 |
| corals | 0.694 | 0.497 | 0.793 | 0.590 |

The pattern is consistent and physically sensible: large, distinctive, high-contrast animals are easy; small benthic organisms that sit camouflaged against the seabed in dense groups are hard.

**A note on reading mAP@0.5.** It is *not* "83% correct". It is the mean across classes of the area under the precision–recall curve, counting a detection as correct when its box overlaps the true one by at least half. The two directly interpretable numbers are recall 0.756 — it finds roughly three-quarters of the animals present — and precision 0.839 — roughly five in six of the boxes it draws are real.

---

## 4. The research journey

Seven sessions, each with a hypothesis and a measured outcome. The negative results are included because they are the majority of the record and the most informative part of it.

### Session 1 — the attention bug, the detector, and two broken measurements
Found that the physics-guided attention had no Q/K/V projections and proved by falsification that it could not perform the colour shift it existed for. Built the RUOD detector to replace the COCO-relabeling approach. Also found that the evaluation scripts scored all 890 UIEB pairs including the ~801 used for training, inflating reported PSNR by a measured **+2.13 dB** (27.24 → 25.11 honest), and that the UCIQE metric implementation was returning values in the **millions** (uncentred LAB chroma plus an unguarded divide-by-luminance; an all-black frame scored 4.66 × 10¹¹).

> **Result:** the fix worked mechanically. The first retrain reached 24.96 dB, below the 25.11 dB baseline, but the run was time-boxed at 50 epochs and still improving.

### Session 2 — the fair comparison, and a falsified prediction
Gave the fixed architecture a full converged budget: 101 epochs to a clean early stop.

> **Result: 24.902 dB — 0.212 dB *below* the pre-fix baseline.** Session 1's "it just needs more epochs" prediction was falsified. Making the headline novelty actually function did not, by itself, improve PSNR. The likeliest reason: the network has several other learnable colour-remapping paths that were already doing that work.

Also ran the enhancement→detection ablation (Section 6).

### Session 3 — reviewing the *rest* of the architecture
The first two sessions fixed the one bug with a visible symptom and then experimented around whatever else was there. Session 3 reviewed every remaining module on design merit and found four more real defects: the physics encoder contained no physics; 65% of parameters served 2.7% of the error; the transformer blocks are exactly permutation-equivariant with no positional encoding anywhere (verified to 1.4 × 10⁻⁷); and `L_frequency` was 99.97% correlated with pixel L1 — a duplicate term.

> **Result: 25.364 dB, the first configuration to beat the baseline — with 15.4% fewer parameters.** Six changes were bundled, so which of them caused it remains unattributed.

### Session 4 — chasing the +3.2 dB, and finding it was never there
The colour-correction module was underused and an oracle test said +3.36 dB was available. A dedicated loss term was added to force it, calibrated by measuring the gradient at the module itself.

> **Result: worse — 24.982 dB.** And the reason, measured *before* the run: the oracle uses the ground-truth reference, and the required offset is only weakly predictable from the input (held-out R² 0.015 / 0.104 / 0.346). Only **24.4%** of the apparent headroom is reachable by any input-conditioned method. The "+3.2 dB of headroom" claim was retired.

### Session 5 — dataset feasibility
A verification-only pass established that LSUI (4,279 pairs) and EUVP are fetchable without authentication, and — critically — that neither contains any of the 89 held-out UIEB test images. The check used perceptual hashing validated against a positive control first: it correctly identifies a genuine duplicate at 1-bit Hamming distance and 0.995 correlation, and found none against the held-out set.

### Session 6 — more data
Trained on UIEB + LSUI (5,080 pairs, 6.3× the data) at matched gradient steps, then fine-tuned back to UIEB alone to correct for LSUI's warmer reference convention.

> **Result: 25.334 dB — 0.031 dB below session 3.** More data did not help. And the risk the two-stage design existed to prevent **never materialised**: the union-trained model's colour calibration was *better* than the UIEB-only model's (R/B within 0.001 of target versus 0.023). The mitigation solved a problem that was not happening.

### Session 7 — honest positioning
Literature review, this report, the website and the PDF. Section 7 is its output.

---

## 5. Results

All enhancement figures are on the same 89 held-out UIEB images, evaluated identically, with the split asserted unchanged across every session.

| Model | Training pairs | PSNR | SSIM | Parameters |
|---|---|---|---|---|
| Pre-fix baseline | 801 | 25.114 dB | 0.9281 | 2,729,450 |
| Post-fix, time-boxed (S1) | 801 | 24.956 dB | 0.9261 | 2,729,450 |
| Post-fix, converged (S2) | 801 | 24.902 dB | 0.9267 | 2,729,450 |
| **Architecture review (S3) — current best** | **801** | **25.364 dB** | **0.9289** | **2,308,723** |
| + forced colour loss (S4) | 801 | 24.982 dB | 0.9240 | 2,308,723 |
| LSUI union pretrain (S6) | 5,080 | 24.806 dB | 0.9177 | 2,308,723 |
| Union + UIEB fine-tune (S6) | 5,080 → 801 | 25.334 dB | 0.9262 | 2,308,723 |

**Detection:** mAP@0.5 = 0.8292, mAP@0.5:0.95 = 0.5845, on 4,200 held-out RUOD images.

### See it yourself
- **[Enhancement visual proof](../outputs/session4_proof.html)** — raw / baseline / current model / reference, on held-out images, including the ones that got worse.
- **[Detection visual proof](../outputs/detection_proof.html)** — predicted boxes beside human-marked ground truth, covering the detector's best *and* worst classes.

### Benchmark context — and why our number is not directly comparable

| Method | Venue | UIEB split | PSNR | SSIM |
|---|---|---|---|---|
| WaterNet | TIP 2019 | Test-U90 | 19.81 | 0.86 |
| FUnIE-GAN | RA-L 2020 | Test-U90 | 19.45 | 0.85 |
| UGAN | ICRA 2018 | Test-U90 | 20.68 | 0.84 |
| Ucolor | TIP 2021 | Test-U90 | 20.78 | 0.87 |
| U-shape Transformer | TIP 2023 | Test-U90 | 22.91 | 0.91 |
| PCAFA-Net | Sensors 2025 | UIEB | 22.80 | 0.890 |
| *This project* | — | *own seed-42 split, 89 images* | *25.364* | *0.9289* |

**This must not be read as "we beat U-shape Transformer."** Our 89 held-out images are a random seed-42 split of UIEB's 890 pairs chosen by this project; the published numbers use the community's standard Test-U90 split, a *different set of images*. UIEB frames vary enormously in difficulty, and a 2.3 M-parameter model outscoring a published transformer by 2.4 dB is far more likely to reflect an easier split than a better method. Evaluating on the standard split is required before any comparison can be claimed, and this project has not done it.

---

## 6. Does enhancement help detection? No — and this reproduces a known result

The natural assumption for an AUV pipeline is that cleaning up the image first should help the detector. It does not.

**Deployed pipeline** (full 4,200-image held-out set), detector trained on raw frames:

| Input to detector | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|
| Raw frames | 0.8292 | 0.5845 |
| Enhanced frames | 0.7906 | 0.5470 |
| **Change** | **−0.0386** | **−0.0375** |

The obvious explanation — that enhancement smooths away the fine texture small seafloor animals depend on — was tested and **falsified**. Across 300 paired frames, enhanced images are *sharper*: Laplacian variance +18.1%, high-pass energy +39.2%, global contrast +39.8%.

That left domain shift, so a matched-domain control was run: two detectors fine-tuned from scratch on identical budgets, one on raw frames and one on enhanced, each evaluated in its own domain.

| Component | mAP@0.5 |
|---|---|
| Total deployed-pipeline loss | −0.0386 |
| …of which train/test domain shift (recoverable) | ≈ −0.0206 |
| …of which genuine residual cost | **−0.0180** |

About half the damage is a fixable wiring problem; the other half is intrinsic. Enhancement makes frames look better to a person and measurably raises contrast and high-frequency energy, but adds no information the detector can use while perturbing the cues it calibrated on.

**This is a reproduction, not a discovery.** Awad et al. (2024) ran the larger version of this study — nine enhancement models, two datasets, three detectors, including retraining detectors on enhanced data — and reached the same conclusion first. This project's contribution here is a careful single-model replication whose harness self-validated (the raw arm reproduces the detector's own training-time mAP exactly), plus the actionable consequence: `infer_detection.py` now detects on raw frames by default and uses enhancement only for the human-facing view, recovering the full 3.9 mAP points.

---

## 7. Novelty assessment — stated plainly

> **None of this project's architectural ideas are novel.**

**This is a claim about who published first, not about copying.** Nothing below was consulted
during this project's seven sessions of development — the codebase was scaffolded
independently and rebuilt from its own debugging, and this literature search happened only now,
at the end, specifically to check honestly whether the result was original. It was not, but
that reflects a well-trodden design space that several teams reached independently, not
derivation from any of the work cited below.

**Physics-guided attention is not new.** Sánchez-Ferreira et al. encode a physical prior as "a spatial bias matrix that directly modulates attention affinity" — the same mechanism class as this project's `Softmax(QKᵀ/√d + λP)V`, for underwater deblurring. PCAFA-Net, PGANet, SFormer, physical-guided transformer interaction, and physics-aware diffusion transformers all occupy the same design space. `docs/math.md` calls this "the core novelty of the proposed method"; **that claim is not supportable and should be rewritten.**

**Wavelet/frequency-split transformers for underwater enhancement are not new.** MixRformer is a dual-branch wavelet-domain underwater enhancement network — structurally the same idea as this one. U-ENHANCE, WEDM, WWE-UIE and a Mamba spectral-attentive wavelet network all combine wavelet decomposition with attention for this exact task. It is standard practice in the subfield.

**The enhancement-hurts-detection finding is not new.** Awad et al. published the larger version first, as Section 6 states.

### So what does this project contribute?

Three things survive scrutiny, and all of them are engineering and methodology rather than architecture.

**1. A diagnosed and proven correctness fix.** The attention module could not do what the architecture claimed, and that was established by falsification — MSE 9.000 = 3.0² with the output mean unmoved — not by inspection. The contribution is the diagnosis and the verification methodology; the fixed module itself is textbook multi-head attention. As a demonstration of how to check that a claimed mechanism actually functions, it is worth showing.

**2. A set of controlled negative results.** Each contradicted a prior expectation, several of them the author's own: fixing the attention did not improve PSNR; forcing the colour module made it worse; 6.3× more data did not help; and the predicted colour-style mismatch from mixing datasets never occurred. Negative results with clean controls are genuinely useful for deciding where to spend the next effort. They are not publishable alone.

**3. One item flagged but explicitly not claimed.** The oracle-versus-achievable decomposition from Session 4 — that only 24.4% of an apparent metric headroom is reachable, because the benchmark's references are hand-retouched — is an observation about the *benchmark* rather than the model, and was not found in the searches run here. That is weak evidence of novelty, not strong, and it is recorded as needing a proper literature check rather than presented as a result.

**A project can be legitimate without being novel.** A working, honestly verified implementation with real ablations and a clear account of what does and does not work is a defensible thing to present. Overselling it as new would not be.

---

## 8. Limitations

**The comparison to published methods is not valid.** Our held-out split is our own; published numbers use the standard Test-U90 split. Until the model is evaluated on the standard split, Section 5's table is context and nothing more.

**Session 3's six changes remain unattributed.** They were bundled into one training run, and the +0.250 dB gain has never been traced to any one of them. Three ablation arms would settle it at roughly 3.8 hours each; it has not been done.

**The colour-correction ceiling is real and low.** Only about a quarter of the apparent headroom is reachable from the input. This is a property of the benchmark's hand-retouched references, and it bounds what any input-conditioned method can achieve on this metric.

**UIQM is reported but unverified.** The implementation returns ~10.0 against the ~2–5 usually quoted for UIEB. It is not numerically broken — no blow-ups, no degenerate values — and it was deliberately left alone to preserve comparability with earlier logged runs, but the convention should be confirmed before any publication.

**Dataset size, even after the expansion attempt.** 801 training pairs is small for a transformer. Adding 4,279 LSUI pairs at matched gradient steps did not help, and the union arm had not converged when its budget ran out, so "more data does not help this architecture" is supported but not settled.

**The transformer blocks have no spatial structure.** They are exactly permutation-equivariant, with no positional encoding anywhere in the repository. The high-frequency branch is assigned edge and texture restoration, which is a spatial arrangement problem that a permutation-equivariant operator cannot represent. This was identified but not fixed.

**Compute.** Every session ran on a laptop RTX 4060 under a firmware sustained-power clamp that holds the GPU at roughly a quarter of its power budget after two to three epochs of continuous load, regardless of Windows power settings or which applications are running. This was the binding constraint on how many experiments each session could run, and it is why several runs were shortened.

---

## 9. References

1. Li, C., Guo, C., Ren, W., Cong, R., Hou, J., Kwong, S., Tao, D. *An Underwater Image Enhancement Benchmark Dataset and Beyond* (UIEB / WaterNet). IEEE TIP, 2020. [arXiv:1901.05495](https://arxiv.org/abs/1901.05495)
2. Peng, L., Zhu, C., Bian, L. *U-shape Transformer for Underwater Image Enhancement* (LSUI dataset). IEEE TIP, 2023. [arXiv:2111.11843](https://arxiv.org/abs/2111.11843)
3. Islam, M. J., Xia, Y., Sattar, J. *Fast Underwater Image Enhancement for Improved Visual Perception* (EUVP / FUnIE-GAN). IEEE RA-L, 2020. [arXiv:1903.09766](https://arxiv.org/abs/1903.09766)
4. Fu, C., Liu, R., Fan, X., et al. *Rethinking General Underwater Object Detection* (RUOD dataset). 2023. [arXiv:2206.05970](https://arxiv.org/abs/2206.05970)
5. Awad, A., Saleem, A., Paheding, S., Lucas, E., Al-Ratrout, S., Havens, T. C. *Beneath the Surface: The Role of Underwater Image Enhancement in Object Detection*. 2024. [arXiv:2411.14626](https://arxiv.org/abs/2411.14626)
6. Cheng, K., Zhao, L., Xue, X., Liu, J., Li, H., Liu, H. *PCAFA-Net: A Physically Guided Network for Underwater Image Enhancement with Frequency–Spatial Attention*. Sensors, 2025. [PMC11946397](https://pmc.ncbi.nlm.nih.gov/articles/PMC11946397/)
7. Sánchez-Ferreira, C., et al. *Physically Guided Attention Mechanism for Underwater Motion Deblurring via Cepstrum-Based Blur Estimation*. Journal of Imaging, 12(5):186. [doi:10.3390/jimaging12050186](https://doi.org/10.3390/jimaging12050186)
8. Tian, X., Lei, Y., Zhang, X., Li, Z., Pun, C.-M., Chen, X. *SFormer: SNR-guided Transformer for Underwater Image Enhancement from the Frequency Domain*. 2025. [arXiv:2508.18664](https://arxiv.org/abs/2508.18664)
9. *MixRformer: Dual-Branch Network for Underwater Image Enhancement in Wavelet Domain*. Sensors, 25(11):3302, 2025. [doi:10.3390/s25113302](https://doi.org/10.3390/s25113302)
10. Guan, M., et al. *Learning A Physical-aware Diffusion Model Based on Transformer for Underwater Image Enhancement*. 2024. [arXiv:2403.01497](https://arxiv.org/abs/2403.01497)
11. Liu, Q., et al. *Degradation-Aware Self-Attention Based Transformer for Blind Image Super-Resolution*. 2023. [arXiv:2310.04180](https://arxiv.org/abs/2310.04180)
12. *Prior-guided attention network for underwater image enhancement*. Computers & Electrical Engineering, 2025. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0045790625003040)
13. *Revisiting Underwater Image Enhancement for Object Detection: A Unified Quality–Detection Evaluation Framework*. Journal of Imaging, 2026. [doi:10.3390/jimaging12010018](https://doi.org/10.3390/jimaging12010018)
14. Chen, X., et al. *Joint Perceptual Learning for Enhancement and Object Detection in Underwater Scenarios*. 2023. [arXiv:2307.03536](https://arxiv.org/abs/2307.03536)
15. Khanam, R., Hussain, M. *YOLOv11: An Overview of the Key Architectural Enhancements*. 2024. [arXiv:2410.17725](https://arxiv.org/abs/2410.17725)
