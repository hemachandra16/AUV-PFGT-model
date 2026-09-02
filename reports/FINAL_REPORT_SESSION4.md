# PFGT-UIE — Session 4: the +3.2 dB was never there to claim

**2026-08-22, ~00:15 → ~04:15 local · unattended (bypass-permissions).**

> Prior reports: [session 1](FINAL_REPORT.md) · [session 2](FINAL_REPORT_150EPOCH.md) ·
> [session 3](FINAL_REPORT_SESSION3.md). Full log: [`PROGRESS.md`](PROGRESS.md).
> **Look at the pictures instead:** [`outputs/session4_proof.html`](../outputs/session4_proof.html)
> — double-click it, no setup needed.

---

## The short version

**Tonight's change did not work, and that turned out to be the useful part.**

The plan was to force the model to use a colour-correction component that session 3 built but
which was barely doing anything — worth "+3.2 dB" by an earlier measurement. I added a loss
term that supervises exactly that, verified it was actually driving the component, and trained.

Result: **PSNR went down, from 25.364 to 24.982.** The "+3.2 dB" headroom barely moved.

The reason is the finding of the night, and I measured it *before* the training run rather
than discovering it afterwards: **most of that +3.2 dB was never achievable.** It was an
*oracle* number — computed using the correct answer. When you check how much of it a model
could reach from the input photograph alone, it is about a quarter, and the rest is a human
retoucher's taste rather than anything recoverable from the image.

So the honest state of the project:

* **Best model is still session 3's, at 25.364 dB.** It has been restored as
  `checkpoints/best.pt`, and `configs/train.yaml` was set back to match it.
* **The "+3.2 dB of headroom" claim — which I made in session 3 — should be retired.** The real
  ceiling for that mechanism is about +0.68 dB, and session 3 already captured +0.14 of it.
* **~0.55 dB is still theoretically available**, but not by the route tried tonight.

Nobody needs to chase this particular avenue again. That is worth more than another 0.1 dB.

---

## 1. The comparison table

| Model | Epochs | Params | PSNR (held-out 89) | SSIM |
|---|---|---|---|---|
| Pre-fix baseline | 115 | 2,729,450 | 25.114 dB | 0.9281 |
| Session 1 (time-boxed) | 50 | 2,729,450 | 24.956 dB | 0.9261 |
| Session 2 (converged) | 101 | 2,729,450 | 24.902 dB | 0.9267 |
| **Session 3 — best, and the one installed** | **96 (best@76)** | **2,308,723** | **25.364 dB** | **0.9289** |
| Session 4 (this run, `L_dc` added) | 75 (best@55) | 2,308,723 | 24.982 dB | 0.9240 |

Session 4 converged on its own terms (early stopping at epoch 75), watchdog logged **0 restarts
and 0 incidents**. It is a clean run that produced a worse model.

---

## 2. What was tried, and how it was verified before spending the GPU time

`L_dc = |mean(prediction, per-image per-channel) − mean(target, …)|`, computed exactly as
`tools/_oracle_dc.py` computes its oracle target so loss and measurement are directly
comparable.

**L1 rather than MSE**, on measured grounds: the DC errors are ~0.035, and `d|x|/dx = 1` while
`d(x²)/dx = 2x = 0.070` there — MSE would push about **14× more weakly** exactly where the push
was needed.

**`lambda_dc = 1.0`, calibrated by gradient rather than guessed.** The brief suggested 0.1–0.2.
Measuring the gradient *at the module* shows that would have reproduced session 3's failure:

```
||grad|| at global_correction.predictor from L_dc @ weight 1.0 : 3.45e-01
||grad|| at the same params from L1 + SSIM + perceptual        : 2.33e-01

 lambda_dc   ratio vs others   % of objective
       0.3             0.44x             6.6%   <- L_dc still LOSES at the module
       1.0             1.48x            19.0%   <- chosen
parity at lambda_dc = 0.68
```

All three pre-training checks passed: the gradient reached the module, dominated there at
1.48×, and optimising it reduced the DC error while quadrupling the per-image variation.

**The fix was correctly built and correctly wired. It still did not help.** That is the point.

---

## 3. Why it did not work — measured, not guessed

This took three hypotheses, two of which I killed with measurements:

1. **"The context cannot predict the offset."** Ordinary least squares said it could:
   R² = 0.692 / 0.866 / 0.897. Apparently refuted.
2. **"The clamp inside the module is eating the gradient."** Measured: only **0.93%** of pixels
   are clamped, and removing it changed the fit from 18.4% to 16.5%. Dead.
3. **Then I distrusted my own R².** That fit put **64 context dimensions through 89 samples** —
   it could not not overfit. Redone with ridge regression fit on the 801 training images and
   evaluated on the held-out 89:

```
                                          R2_R    R2_G    R2_B
physics_context, HELD-OUT                0.015   0.104   0.346
physics_context, in-sample (my first)    0.670   0.847   0.886   <- overfitting, not evidence
```

**The offset is mostly not predictable from the input photograph.** The oracle computes it
from the ground-truth reference, and UIEB's references are *human-retouched* — so a large part
of that "headroom" is a retoucher's decision, not a property of the photograph.

`tools/_achievable_ceiling.py` puts a number on it. Best linear offset predictor, fit on the
801 training images, applied to the held-out 89:

| condition | PSNR | vs base |
|---|---|---|
| model with the correction removed | 22.594 | — |
| + CONSTANT offset (**what session 3 learned**) | 22.730 | +0.136 |
| + offset PREDICTED from the input (held-out) | 23.275 | **+0.681** |
| + ORACLE offset (uses the ground truth) | 25.387 | +2.793 |

**Only 24.4% of the headroom is reachable at all.**

---

## 4. What the training run then showed

| measure | session 3 | session 4 | change |
|---|---|---|---|
| held-out PSNR | 25.364 | 24.982 | **−0.382** |
| held-out SSIM | 0.9289 | 0.9240 | −0.0049 |
| oracle headroom remaining | +3.358 | +3.192 | −0.166 |
| DC share of remaining error | 45.2% | 44.4% | −0.8 pts |
| predicted gain std (per-image variation) | [.022, .025, .033] | [.022, .017, .037] | ~unchanged |

Read together: `L_dc` bought a **0.166 dB** reduction in headroom and cost **0.382 dB** of
PSNR. The 19% of the objective it took came out of the spatial terms, and the module still
converged to roughly the same near-constant correction — because a stronger loss cannot create
information the input does not contain.

This directly answers the brief's question, which asked for both numbers: **PSNR fell, and the
headroom did not close.** Not "PSNR improved by another mechanism", not "headroom closed with
diminishing returns" — neither. The intervention was net negative.

---

## 5. Ablation of session 3's six changes — not done, and why

Each arm is a full training run. The GPU spent tonight power-throttled to ~19 W of 77 W
(`ChatGPT.exe` was a co-client again; per standing instruction I re-applied the power-plan fix
and left the user's application alone), which put epochs at ~146 s instead of ~50 s. One arm is
therefore ~3 hours, and there was not room for even a single one after the main run.

It remains the highest-value next experiment — see below.

---

## 6. What I would do next, in order

1. **Ablate session 3's six changes.** Still unanswered: which of them produced the +0.250 dB.
   Toggle one at a time (`use_global_correction`, `use_physics_priors`, `high_mlp_ratio`,
   GroupNorm, the second fusion conv, `lambda_frequency`). At full clock each arm is ~1.5 h.
   Do this **before** any new idea — the project has now had two sessions of bundled changes
   with no attribution.
2. **Fix the throttling properly.** Every session has lost roughly 3× throughput to it. Worth
   half an hour with the OEM power utility, or simply closing the ChatGPT desktop app while
   training.
3. **If the colour pathway is revisited, change the target, not the pressure.** The measured
   ceiling for input-predicted global colour is +0.68 dB and session 3 holds +0.14 of it. The
   remaining ~0.55 dB would need a *better predictor*, not a stronger loss — e.g. a small
   auxiliary head trained to regress the reference's channel means directly, evaluated by
   held-out R² rather than by training loss.
4. **Consider whether PSNR is the right target at all.** 45% of the remaining error is a global
   colour constant that the ground truth itself only weakly determines. That is a property of
   the *benchmark*, not the model. UIQM/UCIQE, or a no-reference perceptual measure, may be
   more honest for this task.
5. **Still open from earlier sessions:** UIQM reads ~10 against the ~2–5 usually quoted for
   UIEB — not broken, but confirm the convention before publishing.

---

## 7. State of the repository

* `checkpoints/best.pt` — **session 3's model, 25.364 dB**, restored and re-validated after the
  session-4 run proved worse.
* `checkpoints/_session4_result/` — tonight's model (24.982 dB), preserved.
* `checkpoints/_session4_backup/`, `_session3_backup/`, `_baseline_before_fixes/` — every prior
  run, intact.
* `configs/train.yaml` — `lambda_dc: 0.0`, so the config reproduces the installed checkpoint.
  The `_dc_loss` implementation is left in `models/loss.py`, working and documented, in case a
  future experiment wants it.

| Reproduce | Command |
|---|---|
| Held-out metrics | `python validate.py --checkpoint checkpoints/best.pt` |
| The headroom measurement | `python tools/_oracle_dc.py` |
| Predictability (the key result) | `python tools/_ctx_cv.py` |
| The achievable ceiling | `python tools/_achievable_ceiling.py` |
| `L_dc` verification suite | `python tools/verify_dc_loss.py` |
| Rebuild the visual proof | `python tools/make_proof_html.py` |
