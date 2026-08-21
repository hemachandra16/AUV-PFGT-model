# PFGT-UIE — Architecture Design Review, and the First Run That Beats the Baseline

**Session 3 · 2026-08-21, ~18:15 → ~23:00 local · fully unattended (bypass-permissions).**
**Machine:** Ryzen 7 7840HS · 16 GB RAM · RTX 4060 Laptop 8 GB (sm_89).

> Prior context: [`FINAL_REPORT.md`](FINAL_REPORT.md) (session 1 — attention fix, detector build)
> and [`FINAL_REPORT_150EPOCH.md`](FINAL_REPORT_150EPOCH.md) (session 2 — fair comparison,
> detection ablation). Full log and decision record for all three sessions: [`PROGRESS.md`](PROGRESS.md).

---

## 0. Verdict

**Yes — reviewing the rest of the architecture closed the gap and reversed it.**

| Model | Epochs | Params | PSNR (held-out 89) | SSIM |
|---|---|---|---|---|
| Pre-fix baseline | 115 | 2,729,450 | 25.114 dB | 0.9281 |
| Post-fix, time-boxed (session 1) | 50 | 2,729,450 | 24.956 dB | 0.9261 |
| Post-fix, converged (session 2) | 101 (best@81) | 2,729,450 | 24.902 dB | 0.9267 |
| **Session 3 (this run)** | **96 (best@76)** | **2,308,723** | **25.364 dB** | **0.9289** |

**+0.250 dB over the baseline, +0.462 dB over session 2 — with 15.4% FEWER parameters.**
That parameter reduction is deliberate: the result cannot be attributed to added capacity.

Two sessions of empirical tuning around the existing architecture never closed a 0.212 dB gap.
One session of reading the modules and asking whether they do what they claim, closed it and
went past. The premise in the brief was right: the attention bug was the one defect with a
*visible symptom*, so it was the one that got found. It was not the only one.

**But the headline finding of the review is still unclaimed, and it is bigger than the gap.**
See §4 — I predicted the largest fix would come from a specific mechanism, and measurement says
it did not. That is reported as a miss, not smoothed over.

---

## 1. What the review found

I read every remaining module, then ran an independent parallel review to avoid anchoring, and
**measured** every load-bearing claim. Two of my own hypotheses were falsified in the process.

### F5 — THE FINDING: the model had no pathway to emit a per-image colour correction

Verified myself (`tools/_oracle_dc.py`) on the session-2 model at 24.9015 dB:

```
+ oracle per-image per-channel OFFSET  -> 28.1038 dB   (+3.202)
+ oracle per-image per-channel AFFINE  -> 30.3572 dB   (+5.456)
43.1% of remaining error energy is a single per-image per-channel constant
```

**A single per-image constant is worth +3.20 dB — fifteen times the gap two sessions had been
chasing.** And nothing could produce one:

* `model.py` applies `InstanceNorm2d(affine=False)` to the refinement head's **input**, zeroing
  its per-image per-channel mean exactly (measured GAP magnitude 7.0e-09), and to the fusion
  output one line after it is computed (discarding 53.8% of that output's energy).
* The refinement head's receptive field is **9×9**, with no global pooling, and every conv in
  it was `bias=False`.
* The physics encoder was all 3×3 convs, so it could not supply the statistic either.

The one stage able to fix the dominant error was handed a tensor with that information deleted,
looking through a 9×9 window.

### F1 — Capacity was allocated backwards

```
LL (colour/illumination)  97.3% of remaining error energy
LH + HL + HH               2.7%

high_freq_transformer  1,774,725 params (65.0%)
low_freq_transformer     198,533 params ( 7.3%)     ratio 8.9x
```

The branch serving 2.7% of the error held nine times the parameters of the branch serving
97.3%. Not a considered choice: `embed_dim*3` for the concatenated bands times a default
`mlp_ratio=4` made the high-branch FFN alone 43.3% of the whole model, while contributing 8.9%
of its own branch's residual magnitude.

### F2 — The transformer block has no spatial structure at all

```
max |block(P(x), P(pf)) - P(block(x, pf))| = 5.96e-07   (relative 1.36e-07)
adjacent-token swap                        = 3.58e-07
positional-encoding references in the repo : 0
```

**Exactly permutation-equivariant** to float32 precision. Per-token LayerNorm, per-token Linear
FFN, set-based attention, key-side-only physics bias. `docs/architecture.md` assigns this branch
"Texture restoration, Edge enhancement" — an edge *is* a spatial arrangement, so the branch is
being asked to do something its structure forbids. **Not fixed this session** (see §5).

### F3 — The "physics encoder" contained no physics

`Conv3x3 → GELU → 2×ResBlock → Conv3x3`, fed the same RGB the wavelet branch already sees. No
`AdaptiveAvgPool2d`, no `Linear` — so image-wide per-channel statistics are unrepresentable at
any depth. Yet those statistics are what predict the answer: eight closed-form priors regress
against the reference's per-channel gain at **R² = 0.24 / 0.49 / 0.62** (R/G/B).

### F4 — `L_frequency` was a duplicate of L1

```
corr(pixel L1, LL band L1)      = +0.9997
corr(pixel L1, mean-of-4-bands) = +0.9914
weighted contribution: 3.3% of the objective
```

The Haar LL band is a local average, so its L1 is 99.97% correlated with pixel L1, and LL's
magnitude dominates the 4-band mean 10–20×. Session 1 added this term untested; it was a second
copy of L1 at weight 0.15. Set to **0**, which also removes session 1's confound.

### Checked and found FINE — this rules things out, which is also a result

* **InstanceNorm does not erase the colour cast in the encoder.** My hypothesis; falsified by
  measurement (features change *more* after the norm: 39.8% vs 27.9%). I had tested the wrong
  normalisation — the two that matter sit immediately before the decoder (F5).
* **Physics guidance really does reach both branches**, as the README claims.
* **The pre-LN wiring is textbook-correct.** Both residuals branch off the un-normalised stream.
* **`fusion_norm` being reused at two call sites is harmless** — `InstanceNorm2d(affine=False)`
  is stateless.
* **Sigmoid saturation is negligible** (mean sigmoid′ 0.187 vs max 0.25) and the missing final
  bias was worth only ~0.09 dB — both suspicions measured and dismissed.

---

## 2. What was changed, and how each was verified

All six checks pass (`tools/verify_session3_fixes.py`).

| # | Fix | Targets | Verified by |
|---|---|---|---|
| 1 | `GlobalColorCorrection` — per-image affine from a physics context vector, identity at init | F5 | checks 3, 4 |
| 2 | Physics encoder rebuilt: 8 closed-form priors + global-pooling context | F3 | checks 1, 2 |
| 3 | `high_mlp_ratio` 4 → 2 | F1 | param count |
| 4 | `lambda_frequency` 0.15 → 0.0 | F4 | measured correlation |
| 5 | BatchNorm → GroupNorm in refinement + fusion; final conv gains a bias | 0.42 dB train/eval gap at batch 8 | check 5 |
| 6 | Fusion residual block gains a second conv | branch previously ended in GELU (asymmetric) | forward pass |

Selected evidence:

```
CHECK 1  R/G prior falls monotonically with cast severity: 0.725 -> 0.580 -> 0.435 -> 0.290
CHECK 2  encoder now contains Linear/global-pool layers: True (old: False)
CHECK 3  global correction at init: max|out-in| = 0.000e+00, gain exactly 1.0, shift exactly 0.0
CHECK 5  refinement head max|train(x) - eval(x)| = 0.000e+00  (BatchNorm gave a nonzero gap)
CHECK 6  gradients reach context_mlp, se, the 11-ch stem, the predictor, and the new bias
```

---

## 3. The training run

Launched fresh from `configs/train.yaml`, watchdog running, early stopping deciding the endpoint.

```
Early stopping triggered: PSNR did not improve for 20 validation rounds.
Training complete. Best PSNR: 25.2139        (best checkpoint: epoch 76)
96 epochs run.  Watchdog: 0 restarts, 0 incidents.
```

It converged — it was not cut short. Held-out fp32 evaluation: **25.3644 dB / 0.9289 SSIM /
UIQM 10.1158 / UCIQE 0.3221**.

On six held-out samples (`outputs/_final_check_session3/`), mean PSNR-vs-reference went
**25.03 → 26.78 dB**, winning on 4 of 6 (+4.7 dB on `372_img_`, +4.1 on `356_img_`) and losing
on 2 (−2.1 on `172_img_`).

**GPU note:** the run started at 53 s/epoch but dropped to ~146 s partway through. The power
plan had *not* reverted (still High performance) and the card was cool (54 °C) — but it
re-entered `SwPowerCap` at ~19 W of 77 W, and `ChatGPT.exe`'s on-device model service had
appeared as a second GPU client. I re-applied the power overlay (partial recovery, 390→690 MHz)
and **deliberately did not kill your application**. The run completed correctly, just slower.

---

## 4. The honest part: the fix I expected to matter most did not

I predicted F5 — the missing global colour pathway — would be the big win. It was not.

Measured on the **new** model:

```
model as-is                                   25.3644 dB
+ oracle per-image per-channel OFFSET         28.7223 dB   (+3.358)
+ oracle per-image per-channel AFFINE         31.1767 dB   (+5.812)
fraction of error energy that is pure per-image DC: 45.2%   (was 43.1%)
```

**The +3.2 dB headroom is still there — essentially untouched.** The module is active and does
vary per image, but look at how it settled:

```
predicted GAIN  R/G/B: mean [1.250, 1.255, 1.258]   std [0.022, 0.025, 0.033]
predicted SHIFT R/G/B: mean [-0.091, -0.093, -0.101] std [0.015, 0.010, 0.015]
```

It learned an almost **constant** tone adjustment (gain ≈1.25, shift ≈−0.09 for nearly every
image), with per-image modulation of only ~2% of the gain. Yet in isolation (check 4) the same
4,550 parameters captured **91.7%** of the oracle headroom when trained directly against it.

So the pathway is expressive enough; the end-to-end training signal does not drive it there. A
plausible reason: L1/SSIM/perceptual are dominated by spatial error, so a small global offset
contributes little gradient, and the rest of the network can reduce the same losses in other
ways. Nothing forces the global term through this module.

**Consequence for attribution:** six changes were bundled into one run, and I have no ablation.
I cannot tell you which produced the +0.250 dB — and the one I *predicted* would dominate
demonstrably did not. The gain most likely came from the mundane ones: GroupNorm removing a
measured 0.42 dB train/eval mismatch, the capacity rebalance, and dropping a redundant loss
term. That is a less satisfying story than the one I expected to write, and it is the accurate one.

---

## 5. Next steps, in order of expected value

1. **Claim the +3.36 dB.** The headroom is measured, reproducible, and the pathway already
   exists and is provably expressive enough. It just needs a training signal. Cheapest
   experiment: add an auxiliary DC loss —
   `L_dc = |mean(pred) − mean(target)|` per image per channel, weight ~0.1 — or supervise the
   predicted gain/shift directly against the per-image oracle for the first N epochs. This is
   the single highest-value experiment available on this project, by a wide margin.
2. **Ablate the six bundled changes.** Turn each off in turn (`use_global_correction`,
   `use_physics_priors`, `high_mlp_ratio`, `lambda_frequency`, GroupNorm) and re-run. At
   ~50 s/epoch un-throttled, each arm is ~1.5 h. Without this, §0's +0.250 dB is a result
   without a cause.
3. **Give the transformer blocks spatial structure (F2).** A depthwise 3×3 conv between the two
   FFN Linears — the Uformer LeFF / Restormer GDFN pattern — costs ~9·hidden_dim parameters and
   removes the permutation equivariance. The high-frequency branch currently cannot represent
   an oriented edge at all.
4. **A non-redundant frequency loss.** Apply `L_frequency` to LH/HL/HH **only** (correlations
   with pixel L1 of 0.26–0.39, i.e. genuine independent signal) and drop LL, whose consistency
   L1 already enforces.
5. **Still open from earlier sessions:** UIQM reports ~10.0 against the ~2–5 usually quoted for
   UIEB — not broken, but confirm the convention before publishing.

---

## 6. Where the evidence lives

| Artefact | Path |
|---|---|
| Session-3 model | `checkpoints/best.pt` (epoch 76) |
| Prior runs, preserved | `checkpoints/_session3_backup/`, `_50epoch_postfix_backup/`, `_baseline_before_fixes/` |
| Fix verification (6/6 PASS) | `python tools/verify_session3_fixes.py` |
| The headline measurement | `python tools/_oracle_dc.py` |
| Permutation-equivariance proof | `python tools/_permcheck.py` |
| Loss redundancy / error decomposition | `tools/_lossdiag.py`, `tools/_errordecomp.py` |
| Physics prior information content | `tools/_physicsprobe.py` |
| Held-out metrics | `results/validation_metrics_session3.csv` |
| Comparison panels | `outputs/_final_check_session3/` |
| Detection default fix | `python tools/verify_detection_default.py` |

**Phase 0 (unrelated to the model):** `infer_detection.py` now detects on **raw** frames by
default, recovering the 3.9 mAP@0.5 that session 2 measured the `enhance→detect` path was
losing. Enhancement remains available for human viewing via `--annotate-on enhanced`, and the
old behaviour via `--detect-on enhanced`. Verified: default matches raw detections 12/12,
enhanced 0/12, mAP@0.5 = 0.8292.
