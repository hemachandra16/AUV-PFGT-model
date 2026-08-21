# PFGT-UIE — Fair-Comparison Retrain & Detection Ablation

**Session 2 · 2026-08-21, ~14:20 → ~18:30 local · fully unattended (bypass-permissions).**
**Machine:** Ryzen 7 7840HS · 16 GB RAM · RTX 4060 Laptop 8 GB (sm_89, Ada) · driver 592.82.

> Prior context: [`FINAL_REPORT.md`](FINAL_REPORT.md) (session 1 — the architecture fix, the
> detector build, the leakage and UCIQE fixes). Full step-by-step log and decision record for
> both sessions: [`PROGRESS.md`](PROGRESS.md).

---

## 0. TL;DR

1. **The GPU throttle fix worked: 2.9× faster.** 19.46 W → 79.36 W, 585 MHz → 2514 MHz,
   **146 s/epoch → 50 s/epoch**. §1.
2. **The overnight run needed zero intervention.** 101 epochs to a clean early stop,
   **0 watchdog restarts, 0 incidents**. §2.
3. **The headline answer is no.** Given a fair, *converged* training budget, the fixed
   architecture scores **24.902 dB vs the pre-fix baseline's 25.114 dB** — 0.212 dB *worse*.
   Session 1's "it just needs more epochs" prediction is **falsified**. §3.
4. **Bonus finding, and the more useful one: enhancement hurts detection.** The repo's
   current `enhance → detect` default costs **3.9 mAP50 points**. About half is a fixable
   wiring problem; half is intrinsic. §4.
5. **Two hypotheses were tested and killed tonight** — "more epochs will close the gap" and
   "enhancement blurs away texture". Both were mine. Both were wrong. §3, §4.2.

---

## 1. The GPU throttle — fixed, and measured

Session 1 declined to touch system power settings unattended; tonight's brief authorised it.

`powercfg /setactive SCHEME_MIN` could not work as written: **only the Balanced scheme
existed** (Windows 11 hides the legacy High Performance scheme) and the shell was **not
elevated**. What worked without admin:

```powershell
powercfg /duplicatescheme 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c   # creates High performance
powercfg /setactive       5670f263-ce63-477a-91c0-522d01182f6d
powercfg /getactivescheme                                        # -> (High performance)
```

Verified by running 150 s of **real training steps** — the cap only engaged under sustained
load, so a short burst would have proved nothing (`tools/gpu_power_probe.py`):

| | Session 1 (Balanced) | Tonight (High performance) | Change |
|---|---|---|---|
| Power draw | 19.46 W | **79.36 W** | **4.1×** |
| SM clock | 585 MHz (18.8% of max) | **2514 MHz (81.0%)** | **4.3×** |
| Epoch time | ~146 s | **~50 s** | **2.9× faster** |
| Temperature | 52 °C | 47 → 77 °C | hotter, never thermally throttled |

> **A subtlety worth keeping.** `SwPowerCap` is *still* reported after the fix (13/14
> samples). It does **not** mean "throttled" — it means "currently limited by the power
> budget", which is the normal state of any GPU running flat out at its TDP. At 79 W the card
> is at ~103% of its 77 W board limit; last night it was at 25%. **Judge on magnitude, not on
> the flag.** My probe's first verdict function got this backwards and printed "STILL
> THROTTLED" against its own data showing a 4× improvement; I fixed the logic rather than
> trusting the label.

**This change persists on your machine.** To revert: `powercfg /setactive 381b4222-f694-41f0-9685-ff5bb260df2e`.

---

## 2. The overnight run — no intervention required

Launched fresh from `configs/train.yaml` (no `--resume`, no `--epochs` override) so early
stopping decided the endpoint, exactly as the 115-epoch baseline was produced.

```
Early stopping triggered: PSNR did not improve for 20 validation rounds.
Training complete. Best PSNR: 24.7587
Best checkpoint: checkpoints/best.pt   (epoch 81, step 8100)
```

**101 epochs run, best at epoch 81, then 20 flat validation rounds.** It converged — it was
not cut short. That is the single most important fact in §3.

### Watchdog: 0 restarts, 0 incidents

`tools/watchdog.py` polled every 60 s all evening, ready to resume from `latest.pt` if the
process died without a completion marker. It never had to.

```json
[{ "event": "finished", "marker": "Training complete.", "restarts_used": 0 }]
```

Testing the watchdog *before* trusting it found two bugs that would have made it useless:

* **False "alive" from launcher wrappers.** A `nohup → .venv shim → python` chain yields
  *three* processes matching `train.py`: `[(7320, 2253MB), (23196, 6MB), (25400, 4MB)]`. Only
  7320 is real. Had it died with a 6 MB wrapper lingering, a naive check would have reported
  "healthy" and suppressed every restart — precisely the failure a watchdog exists to
  prevent. Fixed by requiring ≥200 MB RSS.
* **No stall detection.** Process presence cannot catch a hang (wedged dataloader, stuck CUDA
  call) where the process stays resident and silent. Added a 900 s log-silence trigger that
  kills the process tree and resumes.

I also caught myself running **two watchdogs** after a hot-swap (a `terminate()` didn't take).
Two is worse than none — both would restart training on a crash, giving two runs fighting
over one GPU and one checkpoint file. Fixed with a singleton guard plus
`tools/_proccheck.py`, a read-only census run at every check-in.

One monitoring lesson worth passing on: **`ps -W` on this machine prints no command-line
arguments**, so `ps -W | grep something.py` never matches and any wait-loop built on it exits
instantly. My first completion waiter did exactly that, firing at epoch 19 of 150. All process
checks now go through `psutil`.

---

## 3. The fair comparison — the fix does not pay off

All figures: held-out 89 images, fp32, fixed UCIQE.

| Model | Epochs | PSNR | SSIM | UIQM | UCIQE |
|---|---|---|---|---|---|
| Pre-fix baseline | 115 | **25.114 dB** | **0.9281** | 10.001 | 0.3133 |
| Post-fix, time-boxed (session 1) | 50 | 24.956 dB | 0.9261 | **10.135** | 0.3125 |
| **Post-fix, converged (tonight)** | **101 (best @ 81)** | 24.902 dB | 0.9267 | 9.963 | **0.3144** |

**Verdict: no.** Given a full, fair, converged budget on an un-throttled GPU, the fixed
architecture lands **0.212 dB below** the pre-fix baseline — and fractionally below even its
own 50-epoch version. SSIM is a tie (−0.0014); UCIQE is marginally the best of the three.

**Session 1's prediction is falsified.** That report argued the 50-epoch model trailed only
because it "was still improving when it hit the epoch limit" and that a full budget would
close the gap. It got a full budget, early-stopped on its own terms, and the gap did not
close. I am recording that plainly rather than quietly dropping it.

### Per-image, it is not a uniform regression

`outputs/_final_check_150ep/` — panels of `raw | pre-fix 115ep | post-fix converged | reference`:

```
image             PSNR ref|OLD  PSNR ref|NEW
15094.png                23.83         27.97     <- new much better
356_img_.png             27.44         28.20
372_img_.png             18.66         18.13
290_img_.png             24.53         22.75
172_img_.png             28.84         25.81
708_img_.png             26.87         21.38     <- new much worse
MEAN                     25.03         24.04
```

The aggregate gap averages genuinely divergent per-image outcomes (+4.1 dB to −5.5 dB), not a
flat degradation.

### An important confound, stated up front

The post-fix rows differ from the baseline in **two** ways, not one: the rebuilt attention,
**and** the added `L_frequency` term (`lambda_frequency: 0.15`) that session 1 introduced
because `docs/math.md` §8 specifies it but it was never implemented. So this compares
**codebases**, which is what tonight's brief asked — it does **not** isolate the attention
change. The clean control is one more run at `lambda_frequency: 0.0`, everything else
identical.

### What this does and does not mean

It does **not** overturn session 1. The old attention module was provably incapable of a
global colour shift (`tools/verify_attention.py`: MSE 9.000 = 3.0², output mean unmoved), and
that is still true. What tonight establishes is narrower and more useful:

> **Making the project's headline novelty actually function does not, by itself, improve PSNR
> on UIEB.**

The likeliest reason: the network has several other learnable colour-remapping paths (physics
encoder convolutions, 1×1 band projections, fusion block, refinement head) that were already
carrying that work. The fix is a **correctness** matter — the paper's central claim now
describes what the code does — not a metrics win. That is worth saying to an advisor plainly,
because "we fixed the novelty and the number went down slightly" is a defensible, honest
position; a fabricated improvement is not.

---

## 4. Bonus: does enhancement help downstream detection?

Session 1 flagged this as the highest-value spare-time experiment. It is, and it produced a
clearer result than the main comparison did. **Exploratory — separate from §3.**

### 4.1 The deployed pipeline (full 4,200-image val set)

`infer_detection.py` runs `raw → enhance → detect`, but the detector was fine-tuned on **raw**
RUOD frames.

```
raw      : mAP50 0.8292   mAP50-95 0.5845
enhanced : mAP50 0.7906   mAP50-95 0.5470
delta    : mAP50 -0.0386  mAP50-95 -0.0375
```

*Harness self-check:* the raw arm reproduces the detector's own training-time `mAP50 = 0.8292`
exactly, so the evaluation path is correct.

Per-class AP50 delta was very uneven — small benthic classes collapsed, large distinctive ones
barely moved:

| worst hit | Δ AP50 | | barely affected | Δ AP50 |
|---|---|---|---|---|
| starfish | −0.0945 | | jellyfish | +0.0009 |
| scallop | −0.0873 | | cuttlefish | −0.0031 |
| echinus | −0.0853 | | turtle | −0.0059 |
| holothurian | −0.0777 | | fish | −0.0081 |

### 4.2 My explanation was wrong — measured, not assumed

The obvious reading is that enhancement smooths away the fine texture small seafloor objects
depend on. I measured it across 300 paired frames instead of asserting it:

| metric | raw | enhanced | change |
|---|---|---|---|
| Laplacian variance | 281.6 | 332.5 | **+18.1%** |
| high-pass energy | 74.17 | 103.24 | **+39.2%** |
| global contrast (std) | 37.95 | 53.05 | **+39.8%** |

Enhanced frames are **sharper and higher-contrast**, not blurrier — only 30% of images lost
Laplacian variance. **Hypothesis falsified.** That left domain shift as the explanation, and
made the matched-domain test decisive rather than optional.

### 4.3 Matched-domain control (equal budget, one variable)

Fine-tuned YOLO11n twice from scratch — 3,000 train frames, 20 epochs, same seed, batch and
image size — one arm on raw, one on enhanced, each evaluated on **its own** domain.

```
raw arm      : mAP50 0.5025   mAP50-95 0.2916
enhanced arm : mAP50 0.4845   mAP50-95 0.2778
delta        : mAP50 -0.0180  mAP50-95 -0.0138
```

### 4.4 Finding

**Enhancement does not help underwater detection on RUOD. It hurts, in both conditions.**
Decomposing the deployed pipeline's loss:

| Component | mAP50 |
|---|---|
| Deployed pipeline loss (raw-trained detector, enhanced input) | −0.0386 |
| …of which is train/test **domain shift** (recoverable) | ≈ −0.0206 |
| …of which is a **genuine residual cost** even when matched | **−0.0180** |

Roughly half the damage is a wiring problem fixable today; the other half is intrinsic.
Enhancement makes frames look better to a human and measurably raises contrast and
high-frequency energy — but adds no information the detector can use, while perturbing cues it
had calibrated on.

**Actionable now:** the repo's default `enhance → detect` path costs ~3.9 mAP50 points. Detect
on **raw** frames and use enhancement for human review; or retrain the detector on enhanced
frames to recover about half the loss.

**Caveat:** the matched arms sit near mAP50 ≈ 0.50 (reduced 3,000-image / 20-epoch budget)
rather than the full detector's 0.829. The comparison is controlled — identical budgets, one
variable — but demonstrated at a lower operating point. Repeating at 9,800 images would settle
whether it transfers.

---

## 5. Where everything is

| Artefact | Path |
|---|---|
| This report | `FINAL_REPORT_150EPOCH.md` |
| Session 1 report | `FINAL_REPORT.md` |
| Full log + decisions (both sessions) | `PROGRESS.md` |
| Converged model | `checkpoints/best.pt` (epoch 81) |
| 50-epoch run (session 1), preserved | `checkpoints/_50epoch_postfix_backup/` |
| Pre-fix baseline, preserved | `checkpoints/_baseline_before_fixes/` |
| Held-out metrics | `results/validation_metrics_150ep.csv` |
| Detection ablation part 1 | `results/ablation_enhance_detect.json` |
| Detection ablation part 3 | `results/ablation_matched_domain.json` |
| Comparison panels | `outputs/_final_check_150ep/` |
| Training log / watchdog log | `logs/train.log`, `logs/watchdog.log` |
| GPU throttle probe | `tools/gpu_power_probe.py` |

---

## 6. Next steps

1. **Isolate the attention change.** One run at `lambda_frequency: 0.0`, everything else
   identical, separates the attention rewrite from the added frequency loss (§3). ~1.5 h at
   current GPU speed. This is the one experiment that would let you say precisely what the
   attention fix cost or bought.
2. **Stop enhancing before detection.** Fastest real win available: ~3.9 mAP50 points, no
   training required — change the default in `infer_detection.py`. Retrain the detector on
   enhanced frames only if you need enhancement in that path for another reason.
3. **Confirm §4 at full scale.** Repeat the matched-domain arms at 9,800 images to check the
   −0.018 residual holds at the 0.83 operating point rather than 0.50.
4. **Decide what the paper claims.** The attention fix is defensible as a *correctness* result
   — the implementation now matches `docs/math.md` §5, and the old one provably could not do
   what was claimed. It is not defensible as a PSNR improvement. §4 is arguably the stronger
   contribution: a controlled negative result on enhancement-for-detection, which is a real
   and publishable finding for AUV vision.
5. **Still outstanding from session 1:** UIQM reports ~10.0 against the ~2–5 usually quoted for
   UIEB. Not numerically broken, so it was left alone, but confirm the convention before
   publication.
6. **Not done, deliberately:** stacking 2 `TransformerBlock`s per branch. It changes model
   capacity and would confound item 1. Do it after that, not before.

### What I decided without being able to ask

Recorded as S2-D-001 … S2-D-004 in `PROGRESS.md`. The two worth your review:

* **S2-D-001** — I changed your Windows power plan to High performance. It persists until you
  change it back (§1 has the command). It is the reason tonight's run took 1.4 h instead of 4.
* **S2-D-004** — the §3 comparison is confounded by the added frequency loss. I could have
  spent tonight's spare hours on that control instead of the detection ablation. I judged the
  ablation more valuable because it answers a question about the *project's purpose* rather
  than about attribution between two of its own changes — but that was my call, and item 1
  above exists because of it.
