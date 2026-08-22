# PFGT-UIE — Session 6: does more training data help? No. And not for the reason expected.

**2026-08-22, ~11:40 → ~17:00 local · unattended (bypass-permissions).**

> Prior reports: [S1](FINAL_REPORT.md) · [S2](FINAL_REPORT_150EPOCH.md) ·
> [S3](FINAL_REPORT_SESSION3.md) · [S4](FINAL_REPORT_SESSION4.md) ·
> [feasibility](DATASET_EXPANSION_FEASIBILITY.md). Full log: [`PROGRESS.md`](PROGRESS.md).

---

## The short version

Adding 4,279 LSUI pairs to the 801 UIEB training pair — **6.3× the data, at matched gradient
steps** — did not improve the model. The two-stage recipe recovered almost all of the loss, but
finished **0.031 dB below** the UIEB-only model it was trying to beat. Session 3's checkpoint
remains installed.

**The interesting part is why.** The whole two-stage design existed to defuse one specific risk:
that LSUI's warmer reference colour would drag the model off UIEB's calibration. That risk
**did not materialise at all.** The union-trained model landed on UIEB's colour convention
*more accurately than session 3 did* — R/B within 0.001 of the target, versus session 3's 0.023.

So the mitigation solved a problem that was not happening, and the actual deficit has some other
cause. That is a more useful outcome than a small win would have been, because it rules out the
explanation everyone (including the feasibility report, and me) expected to be the answer.

---

## 1. Results

All figures: held-out 89 UIEB images, fp32, identical evaluation path, split asserted unchanged.

| Model | Training pairs | Steps | PSNR | SSIM | UIQM | UCIQE |
|---|---|---|---|---|---|---|
| Pre-fix baseline | 801 | ~11,500 | 25.114 | 0.9281 | — | — |
| **Session 3 — installed** | **801** | **9,600** | **25.364** | **0.9289** | 10.116 | 0.3221 |
| S6 stage 1 — union | 5,080 | 9,525 | 24.806 | 0.9177 | 9.869 | 0.3126 |
| S6 stage 2 — union + UIEB fine-tune | 5,080 → 801 | 9,525 + 900 | 25.334 | 0.9262 | 10.017 | 0.3138 |

* stage 2 vs session 3: **−0.031 dB, −0.0027 SSIM** — a tie, fractionally worse.
* stage 2 vs stage 1: **+0.528 dB** — the fine-tune did real work.

**Verdict: more data did not help.** Not catastrophically — the final model is within a
rounding error of session 3 — but the 6.3× data increase bought nothing.

### On fairness of the comparison
Stage 1 ran **9,525 gradient steps against session 3's 9,600** — step-parity, deliberately, so
the comparison isolates *data variety* from *compute*. Two caveats stated plainly:

1. Stage 1 ran the full 15 epochs **without early-stopping and was still improving** (last four
   validations 24.01 → 24.38 → 24.43 → 24.47). Its 24.806 is a floor, not a converged value. A
   longer union run might do better; this session could not afford one (§4).
2. Stage 2 early-stopped legitimately at epoch 21, best at epoch 9 — that half is converged.

So the honest claim is narrower than "more data doesn't help": **at equal optimisation budget,
more data did not help, and the union arm had not finished improving.**

---

## 2. The mitigation worked mechanically — and was aimed at the wrong target

The feasibility report measured that LSUI's references are warmer than UIEB's and warned that a
naive union would pull the model's calibration off. Measured over the held-out 89:

| | R/B | distance from UIEB's target |
|---|---|---|
| UIEB raw (input) | 0.523 | — |
| **UIEB reference (the target)** | **0.779** | — |
| Session 3 (UIEB only) | 0.802 | 0.023 |
| **S6 stage 1 (union, no fine-tune)** | **0.780** | **0.001** |
| S6 stage 2 (union + UIEB fine-tune) | 0.771 | 0.007 |
| LSUI GT (the other convention) | 0.857 | — |

**The predicted warm drift did not happen.** Training on the union produced a model whose global
colour calibration is essentially exact — 0.001 from UIEB's target, an order of magnitude better
than session 3's 0.023. If anything the extra data *improved* colour calibration, presumably
because 6.3× more varied underwater scenes is a better sample of "what underwater colour looks
like" than 801 images.

And stage 2's fine-tune moved calibration slightly *away* again (0.001 → 0.007) while raising
PSNR by 0.53 dB. So the fine-tune's benefit came from something other than colour.

The conclusion is uncomfortable but clear: **stage 1's 0.56 dB deficit is not a colour-style
problem.** The two-stage architecture was insurance against a risk that never fired. Plausible
remaining explanations, none tested here:

* LSUI is much lower-resolution (39% at 320×240) than UIEB (up to 1800×1295). Training on
  upsampled low-detail material may teach texture priors that hurt on UIEB's sharper images.
* 15 epochs over a 6.3× pool means each individual UIEB image was seen ~15 times versus ~96 in
  session 3. Step-parity is not exposure-parity per image, and per-image exposure may matter more
  for a 2.3 M-parameter model than total sample count.

The second is the more likely candidate and is directly testable — see §5.

---

## 3. Per-image behaviour

Six held-out images (`outputs/_final_check_session6/`), PSNR vs reference:

```
image             session3    stage1    stage2
708_img_.png         28.35     26.34     26.33
15094.png            26.74     24.79     27.29
356_img_.png         31.56     24.16     26.48
372_img_.png         23.39     28.83     27.88
290_img_.png         23.88     24.35     24.60
172_img_.png         26.73     23.92     25.74
MEAN                 26.78     25.40     26.39
```

Not a uniform regression: the union model **wins by 5.4 dB on `372_img_`** and loses by 7.4 dB
on `356_img_`. The aggregate tie hides genuinely divergent per-image behaviour, which is
consistent with "different training distribution" rather than "uniformly worse model".

---

## 4. Two operational findings worth more than the metric

### ChatGPT.exe is not the throttling root cause
This session's brief stated the GPU throttling had been "traced every time" to ChatGPT.exe.
Ten of its processes were found and force-killed at Phase 0, and that **did** produce the best
power figure any session has recorded — 87.76 W / 2535 MHz, ~340 s/epoch. But the clamp returned
at epoch 3 with ChatGPT.exe absent and the trainer the only GPU compute client:

```
19.33 W / 705 MHz / 55 C / util 99% / SwPowerCap     629 s/epoch, then 953 s
re-applied High performance + OVERLAY_SCHEME_MAX, on AC, card cool -> 19.31 W / 840 MHz
```

Windows power settings maxed, on mains, cool, sole client, still clamped to a quarter of budget.
**The cause is the laptop's firmware sustained-power policy**, which engages after two to three
epochs of continuous load regardless of software. Killing ChatGPT.exe is still worth doing — it
bought full clocks for two epochs — but the project should stop treating it as the fix. A real
fix needs the OEM power utility or a BIOS setting, which is outside what an unattended session
should touch.

This cost the session real capability: the stage-1 horizon had to be cut 30 → 20 → 15 epochs as
epoch time went 340 s → 953 s. At full clock the original 30-epoch plan would have fitted
comfortably, and §1's caveat about stage 1 not having converged would not exist.

### `--resume` would have silently ruined the fine-tune
Stage 2 starts from stage 1's weights, and the obvious way to do that is `--resume`. It would
have quietly destroyed the experiment: `--resume` restores `global_step`, and the LR is a cosine
over `global_step / total_steps`. Stage 1 ends near step 9,525 against a 4,000-step stage-2
schedule, so the cosine would be driven past its end and swing **back up**:

```
step 19050 -> cosine progress 7.4 -> lr 7.58e-05
step 20000 -> cosine progress 7.8 -> lr 1.81e-04     (base_lr 2.00e-04)
```

A "fine-tune" running at 90% of the from-scratch rate would have produced a bad number that
looked like "the mitigation doesn't work" rather than "the LR was wrong". Added `--init-from`
(weights only, fresh optimizer and schedule), mutually exclusive with `--resume`.

---

## 5. What I would do next

1. **Test exposure-parity, not step-parity.** The most likely explanation for stage 1's deficit
   is that each UIEB image was seen ~15 times instead of ~96. Re-run the union for ~90 epochs
   (~57,000 steps) so per-image exposure matches. At full clock that is ~8 h; at the clamped
   rate it is ~24 h and not worth attempting. **Fix the throttling first** — it is now the
   binding constraint on every experiment this project wants to run.
2. **Try LSUI as a curriculum rather than a union.** Pretrain on LSUI *alone*, then fine-tune on
   UIEB. That gives full per-image exposure on both, in sequence, and is cheaper than (1).
3. **Test the resolution hypothesis.** Restrict LSUI to its ≥640 px subset (~35% of it) and
   re-run the union. If the deficit shrinks, low-resolution training material is the cause and
   the fix is filtering, not more epochs.
4. **Do not add EUVP yet.** It was correctly kept out of scope for single-variable cleanliness,
   and there is no evidence yet that more data of *any* kind helps this architecture.
5. **Still open:** the session-3 six-change ablation (which of them produced +0.250 dB) remains
   unanswered from two sessions ago, and UIQM reads ~10 against the ~2–5 usually quoted for UIEB.

---

## 6. Repository state

* `checkpoints/best.pt` — **unchanged, session 3's model at 25.364 dB.** Verified by md5 against
  `_session6_backup/best.pt`; the stage runs wrote to `_stage1/` and `_stage2/` subdirectories
  and never touched it.
* `checkpoints/_stage1_union.pt` (24.806) and `_stage2_finetuned.pt` (25.334) — preserved.
* `configs/train.yaml` — **unchanged**, still reproducing the installed checkpoint. The new
  recipe lives in `train_stage1_union.yaml` / `train_stage2_finetune.yaml` and was deliberately
  not promoted, since it did not win.
* `datasets/LSUI/` — 4,279 pairs laid out and leak-gated; gitignored.

| Reproduce | Command |
|---|---|
| Leak gate (mandatory before any training) | `python tools/check_dataset_leakage.py` |
| Colour-style analysis | `python tools/check_color_style.py` |
| Held-out metrics | `python validate.py --checkpoint <ckpt>` |
| Stage 1 / stage 2 | `train.py --config configs/train_stage1_union.yaml` then `--config configs/train_stage2_finetune.yaml --init-from checkpoints/_stage1_union.pt` |

Watchdog: **0 restarts, 0 crash incidents** across both stages. Both runs ended on their own
terms (stage 1 at its epoch limit, stage 2 by early stopping).
