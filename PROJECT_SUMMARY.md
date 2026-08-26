# PFGT-UIE — project index

**Physics-Guided Frequency Transformer for Underwater Image Enhancement, with marine object
detection.** Underwater photographs lose red light to absorption and contrast to backscatter
before they reach the sensor. This project enhances them, detects ten classes of marine object,
and — over eight sessions — documents what worked, what did not, and what was already published
by someone else.

| | |
|---|---|
| Enhancement, 89 held-out UIEB images | **25.364 dB PSNR · 0.9289 SSIM** |
| Detection, 4,200 held-out RUOD images | **mAP@0.5 = 0.829 · mAP@0.5:0.95 = 0.585** |
| Model size | 2.31 M parameters (enhancer) · 2.6 M (YOLO11n detector) |
| Architectural novelty | **None claimed** — the mechanisms are published prior work |

Three findings are worth knowing before reading anything else:

- **Enhancement makes detection worse, not better** (−3.9 mAP points deployed). The pipeline
  therefore detects on raw frames and uses enhancement only for the human-facing view. This
  reproduces a result Awad et al. published first.
- **No comparison to published work is possible.** There is no standard UIEB test split —
  the dataset's authors never published one, and each paper uses its own random 90 images. The
  one public list that exists is 87.8% inside this project's training set.
- **Nothing here is architecturally new.** Physics-biased attention, wavelet-split transformers
  for underwater enhancement, and the enhancement-hurts-detection result all exist in the
  literature. What this project contributes is a diagnosed correctness fix, a set of clean
  negative results, and an honest account of both.

---

## Start here

| | What it is |
|---|---|
| **[The report — website](outputs/website.html)** | The full write-up: architecture module by module, results, the research journey, the novelty assessment, limitations, references. Self-contained, opens by double-click, light and dark. |
| **[The report — PDF](outputs/PFGT-UIE_report.pdf)** | The same content, 18 pages, for printing or sending. |
| **[Enhancement: see it yourself](outputs/session4_proof.html)** | Held-out photographs: original, enhanced, human reference. Includes the images the model made *worse* — 39 better, 14 about the same, 36 worse. |
| **[Detection: see it yourself](outputs/detection_proof.html)** | Predicted boxes beside human-marked ground truth on ten held-out frames, deliberately covering the detector's three worst classes as well as its three best. |

Both report formats are rendered from one source, [`docs/report_content.md`](docs/report_content.md),
so they cannot disagree about a number. Rebuild either with `python tools/build_website.py` or
`python tools/build_pdf.py`.

---

## The written record

Eight sessions, in order. Each report states its hypothesis and its measured outcome, including
when the outcome contradicted the hypothesis — which was most of the time.

| # | Report | What it found |
|---|---|---|
| 1 | [`FINAL_REPORT.md`](FINAL_REPORT.md) | The physics-guided attention module had **no Q/K/V projections** and was provably incapable of the colour shift it existed for. Also: evaluation was scoring training images, inflating PSNR by 2.13 dB, and the UCIQE implementation returned values in the millions. |
| 2 | [`FINAL_REPORT_150EPOCH.md`](FINAL_REPORT_150EPOCH.md) | Fixing the attention and training to convergence gave **24.902 dB — below the pre-fix baseline.** Session 1's "it just needs more epochs" prediction, falsified. Also ran the enhancement→detection ablation. |
| 3 | [`FINAL_REPORT_SESSION3.md`](FINAL_REPORT_SESSION3.md) | Reviewed every remaining module on design merit and found four more real defects. **25.364 dB with 15.4% fewer parameters** — the first configuration to beat the baseline, and the one still installed. |
| 4 | [`FINAL_REPORT_SESSION4.md`](FINAL_REPORT_SESSION4.md) | Chased an apparent +3.2 dB of colour-correction headroom. Got **24.982 dB — worse** — and then measured that only **24.4% of that headroom was ever reachable**, because the benchmark's references are hand-retouched. The claim was retired. |
| 5 | [`DATASET_EXPANSION_FEASIBILITY.md`](DATASET_EXPANSION_FEASIBILITY.md) | Verification-only: LSUI and EUVP are fetchable, and — checked by perceptual hashing validated against a positive control — **contain none of the 89 held-out test images.** |
| 6 | [`FINAL_REPORT_SESSION6.md`](FINAL_REPORT_SESSION6.md) | 6.3× more training data at matched gradient steps gave **25.334 dB — 0.031 dB below session 3.** More data did not help, and the colour-style mismatch the two-stage recipe existed to prevent **never occurred**. |
| 7 | [`docs/novelty_assessment.md`](docs/novelty_assessment.md) | The literature check. Verdict: **none of this project's architectural ideas are novel**, with citations for each mechanism. |
| 8 | [`FINAL_REPORT_SESSION8.md`](FINAL_REPORT_SESSION8.md) | Went looking for the standard UIEB test split to get a comparable number. **There is no standard split**, and the one published list that exists is 87.8% inside this project's training data. Measured the memorisation gap at **+4.88 dB**. |

Full chronological log, including every decision and its reasoning: [`PROGRESS.md`](PROGRESS.md).

---

## Reference documents

| File | Contents |
|---|---|
| [`docs/report_content.md`](docs/report_content.md) | Canonical report source. Edit this, not the rendered HTML or PDF. |
| [`docs/novelty_assessment.md`](docs/novelty_assessment.md) | Mechanism-by-mechanism comparison against 2023–2026 prior work, with checkable links. |
| [`docs/standard_split_investigation.md`](docs/standard_split_investigation.md) | Why no comparison to published UIEB numbers is available, and what one would cost. |
| [`docs/math.md`](docs/math.md) | The formulation, module by module. |
| [`docs/architecture.md`](docs/architecture.md) | Module and tensor-shape reference. |
| [`docs/literature.md`](docs/literature.md) | Background reading. |
| [`README.md`](README.md) | Setup, dependencies, training and inference commands. |

---

## Reproducing the numbers

```bash
python validate.py --checkpoint checkpoints/best.pt
```

| Task | Command |
|---|---|
| Held-out enhancement metrics | `python validate.py --checkpoint checkpoints/best.pt` |
| Dataset leak gate (run before any training) | `python tools/check_dataset_leakage.py` |
| Attention correctness proof | `python tools/verify_attention.py` |
| Colour-style analysis | `python tools/check_color_style.py` |
| Rebuild the website | `python tools/build_website.py` |
| Rebuild the PDF | `python tools/build_pdf.py` |
| Rebuild both proof pages | `python tools/make_proof_html.py` · `python tools/make_detection_proof.py` |
| Score a checkpoint on an explicit filename list | `python tools/eval_on_list.py --list docs/splits/uieb_T90_ddz16.txt` |

`configs/train.yaml` reproduces the installed checkpoint. The session-6 two-stage recipe lives
in `train_stage1_union.yaml` / `train_stage2_finetune.yaml` and was deliberately **not**
promoted, because it did not win.

---

## Known limitations

Stated in full in section 8 of the report. The four that matter most:

1. **No comparison to published methods is available.** Not because this project used the
   wrong split — because **there is no standard UIEB split.** The dataset's authors publish
   none, and the papers each measure on their own random 90 images, so the published figures are
   not strictly comparable to each other either. The one public filename list that exists
   overlaps this project's training set by 87.8%, so the installed model cannot be scored on it;
   doing so anyway gives an inflated 28.19 dB. See
   [`docs/standard_split_investigation.md`](docs/standard_split_investigation.md).
2. **Session 3's six changes remain unattributed.** They were bundled into one run, and the
   +0.250 dB has never been traced to any one of them.
3. **UIQM is reported but unverified** — the implementation returns ~10.0 against the ~2–5
   usually quoted for UIEB. Not numerically broken, but the convention is unconfirmed.
4. **The transformer blocks have no positional encoding** and are exactly permutation-
   equivariant, which is a poor fit for the high-frequency branch's job. Identified, not fixed.

Compute throughout was a laptop RTX 4060 under a firmware sustained-power clamp that holds it at
roughly a quarter of its power budget after two to three epochs of continuous load. That was the
binding constraint on how much each session could test.
