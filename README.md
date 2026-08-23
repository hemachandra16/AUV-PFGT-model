# Physics-aware Frequency-Guided Transformer for Underwater Image Enhancement (PFGT-UIE)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **PFGT-UIE** is a deep learning architecture for underwater image enhancement that integrates underwater imaging physics directly into transformer attention, rather than treating physics as a pre-processing step.
>
> **On novelty:** this mechanism is *not* new — encoding a physical prior as an additive pre-softmax attention bias is established prior work, and so is the wavelet/frequency split around it. See [`docs/novelty_assessment.md`](docs/novelty_assessment.md) for the citations and the full verdict. What this repository offers is a working, honestly verified implementation with real ablations and a documented account of what did and did not help.

---

## Architecture Overview

```
Input RGB Image (B, 3, H, W)
        │
        ▼
Physics Prior Encoder ──────────────────────────────────┐
(B, 64, H, W)                                           │
        │                                               │ Physics guidance
        ▼                                               │ injected into every
Single-Level Haar Wavelet Transform                     │ transformer block
(LL, LH, HL, HH) each (B, 3, H/2, W/2)                │
        │                                               │
    ┌───┴────┐                                          │
    │        │                                          │
    ▼        ▼                                          │
  LL Band  [LH, HL, HH]  ←──────────────────────────── ┘
    │        │
    ▼        ▼
Low-Freq    High-Freq
Transformer Transformer   ← Physics-Guided Attention
    │        │
    └───┬────┘
        │
        ▼
Cross-Frequency Feature Fusion  +  Physics Features
(128-channel fused representation)
        │
        ▼
Inverse Haar Wavelet Reconstruction
        │
        ▼
Residual Feature Injection
        │
        ▼
Image Refinement Head (CNN decoder)
        │
        ▼
Enhanced RGB Image (B, 3, H, W) ∈ [0, 1]
```

### Physics-Guided Attention

Standard attention: `Softmax(QKᵀ / √d) · V`

**PFGT-UIE attention**: `Softmax(QKᵀ / √d + λP) · V`

where **P** is a learned physics bias derived from the Physics Prior Encoder. This allows the transformer to focus on regions with severe degradation (attenuation, scattering, color distortion).

This is a known mechanism class rather than a contribution of this project — see [`docs/novelty_assessment.md`](docs/novelty_assessment.md). It is worth noting that in the original implementation this module had no Q/K/V projections at all and was provably incapable of the colour shift it existed to perform; the diagnosis and the falsification test are in [`FINAL_REPORT.md`](FINAL_REPORT.md).

---

## Dataset

**UIEB** (Underwater Image Enhancement Benchmark)  
890 paired raw / reference images.

Expected folder structure:
```
datasets/
└── UIEB/
    ├── raw-890/        ← degraded underwater images
    └── reference-890/  ← ground-truth reference images
```

---

## Installation

```bash
# Create and activate a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Install pytorch-wavelets (required for wavelet modules)
pip install pytorch-wavelets
```

---

## Training

```bash
# Full training (150 epochs, batch 4, AMP enabled)
python train.py --config configs/train.yaml

# Quick smoke test (2 epochs, debug mode)
python train.py --config configs/train.yaml --epochs 2 --debug-mode

# Resume from checkpoint
python train.py --config configs/train.yaml --resume checkpoints/latest.pt

# Override batch size and learning rate via CLI
python train.py --config configs/train.yaml --batch-size 8 --lr 5e-5
```

TensorBoard logs are written to `logs/tensorboard/`. Launch with:
```bash
tensorboard --logdir logs/tensorboard
```

---

## Validation

```bash
# Evaluate best checkpoint — PSNR, SSIM, UIQM, UCIQE
python validate.py --checkpoint checkpoints/best.pt

# Save per-image CSV report
python validate.py --checkpoint checkpoints/best.pt --output-csv results/metrics.csv
```

---

## Testing

```bash
# Single image
python test.py --mode single --input datasets/UIEB/raw-890/img_001.png

# Folder of images
python test.py --mode folder --input datasets/UIEB/raw-890/

# Full UIEB dataset with metrics
python test.py --mode dataset --checkpoint checkpoints/best.pt
```

Enhanced images are saved to `outputs/` by default.

---

## Inference

```bash
# Single image (preserves original resolution)
python infer.py --input path/to/underwater.png --output path/to/enhanced.png

# Folder inference
python infer.py --input path/to/raw_folder/ --output-dir path/to/results/

# Force a device (default is "auto")
python infer.py --input path/to/raw_folder/ --output-dir path/to/results/ --device cuda
```

> `infer.py` runs at native resolution, one image at a time (padding to a multiple of
> 16 for the wavelet transform, then cropping back). It has no `--batch-size` flag.

### Enhancement + object detection

```bash
# Enhance, then detect with the RUOD fine-tuned underwater detector
python infer_detection.py --input path/to/raw_folder/ --output-dir outputs/detection

# Compare against the legacy COCO Faster R-CNN path
python infer_detection.py --input path/to/img.png --detector fasterrcnn
```

---

## Repository Structure

```
PhysicsFreqTransformer/
├── configs/
│   └── train.yaml              ← All training hyperparameters
├── data/
│   └── dataset.py              ← UIEBDataset + get_splits() (shared seeded 90/10 split)
├── datasets/
│   ├── UIEB/                   ← Enhancement pairs
│   │   ├── raw-890/
│   │   └── reference-890/
│   └── RUOD_yolo/              ← Underwater detection dataset (YOLO format)
├── docs/
│   ├── architecture.md
│   ├── blueprint.md
│   ├── literature.md
│   ├── math.md
│   └── project_spec.md
├── metrics/
│   ├── psnr.py                 ← Peak Signal-to-Noise Ratio
│   ├── ssim.py                 ← Structural Similarity Index
│   ├── uiqm.py                 ← Underwater IQM (Panetta 2016)
│   └── uciqe.py                ← Underwater Color IQE (Yang 2015)
├── models/
│   ├── attention/
│   │   └── physics_attention.py  ← Physics-Guided Attention
│   ├── fusion.py               ← Cross-frequency feature fusion
│   ├── inverse_wavelet.py      ← Inverse Haar DWT reconstruction
│   ├── build.py                ← build_model(): the ONLY model constructor
│   ├── loss.py                 ← L1 + SSIM + Perceptual (VGG19) + Frequency loss
│   ├── object_detection.py     ← RUOD fine-tuned YOLO detector (+ legacy COCO path)
│   ├── model.py                ← Full PFGT-UIE pipeline
│   ├── physics_encoder.py      ← Physics Prior Encoder (CNN)
│   ├── refinement.py           ← Image Refinement Head
│   ├── transformer_block.py    ← Pre-LN Transformer Block
│   └── wavelet.py              ← Single-level Haar DWT
├── utils/
│   ├── checkpoint.py           ← best.pt + latest.pt save/load
│   ├── logging_utils.py        ← Structured logger (console + file)
│   └── seed.py                 ← seed_everything()
├── checkpoints/                ← Saved model weights
├── logs/                       ← Training logs + TensorBoard events
├── outputs/                    ← Enhanced image outputs
├── results/                    ← Metric CSV reports
├── train.py                    ← Training script
├── validate.py                 ← Validation script
├── evaluate.py                 ← Evaluation entry-point
├── test.py                     ← Test script (single/folder/dataset)
├── infer.py                    ← Native-resolution inference (one image at a time)
├── infer_detection.py          ← Enhancement + underwater object detection
├── tools/
│   ├── verify_attention.py     ← Proves the physics-guided attention actually works
│   ├── compare_before_after.py ← Pre-fix vs post-fix image/metric comparison
│   ├── fetch_ruod.py           ← Download the RUOD detection dataset
│   ├── ruod_to_yolo.py         ← COCO -> YOLO conversion for RUOD
│   └── train_detector.py       ← Fine-tune the underwater YOLO detector
└── requirements.txt
```

---

## Configuration

All hyperparameters live in [`configs/train.yaml`](configs/train.yaml):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `training.epochs` | 150 | Total training epochs |
| `dataloader.batch_size` | 8 | Training batch size (peaks ~5.2 GB VRAM at 256x256) |
| `model.num_heads` | 4 | Attention heads (all scripts read this via `models/build.py`) |
| `optimizer.lr` | 2e-4 | AdamW learning rate |
| `scheduler.warmup_epochs` | 5 | Linear warmup duration |
| `training.amp` | true | Automatic Mixed Precision |
| `training.grad_clip_norm` | 1.0 | Max gradient norm |
| `loss.lambda_l1` | 1.0 | L1 loss weight |
| `loss.lambda_ssim` | 0.5 | SSIM loss weight |
| `loss.lambda_perceptual` | 0.1 | Perceptual (VGG19) loss weight |
| `loss.lambda_frequency` | 0.15 | Haar-DWT sub-band consistency loss weight |
| `early_stopping.metric` | psnr | Monitored metric (`psnr` or `ssim`) |
| `early_stopping.patience` | 20 | Validation rounds without improvement |
| `training.save_every_epochs` | 10 | Also write `checkpoints/epoch_N.pt` snapshots |

---

## Reproducibility

All training runs set a fixed seed via `utils/seed.py` (`seed=42` by default):
- Python `random`
- `PYTHONHASHSEED`
- NumPy
- PyTorch (CPU + CUDA)
- cuDNN deterministic mode

Override with `--seed <value>`.

---

## Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| **PSNR** | Peak Signal-to-Noise Ratio | Higher is better (dB) |
| **SSIM** | Structural Similarity Index | [0, 1], higher is better |
| **UIQM** | Underwater Image Quality Measure (Panetta 2016) | Higher is better |
| **UCIQE** | Underwater Color Image Quality Evaluation (Yang 2015) | Higher is better |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{pfgt_uie_2024,
  title   = {Physics-aware Frequency-Guided Transformer for Underwater Image Enhancement},
  author  = {Your Name},
  year    = {2024},
  note    = {https://github.com/yourname/PhysicsFreqTransformer}
}
```

---

## References

- Wang, Z. et al. (2004). "Image quality assessment: from error visibility to structural similarity." *IEEE TIP*.
- Panetta, K. et al. (2016). "Human-visual-system-inspired underwater image quality measures." *IEEE JOE*.
- Yang, M. & Sowmya, A. (2015). "An underwater color image quality evaluation metric." *IEEE TIP*.
- Liu, Z. et al. (2021). "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." *ICCV*.

---

## Future Work

- Multi-scale wavelet processing (2–3 levels)
- Larger VGG perceptual layers for richer semantic loss
- Experiment on other underwater datasets (UOBD, EUVP, U45)
- Knowledge distillation for real-time AUV deployment
