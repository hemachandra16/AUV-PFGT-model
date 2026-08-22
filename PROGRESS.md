# PFGT-UIE — Autonomous Fix + Detection + Overnight Training Run

**Session start:** 2026-08-20 ~00:15 local — **ALL PHASES COMPLETE ~04:50**
**Operator:** unattended (bypass-permissions). All decisions made autonomously and logged here.
**Machine:** Ryzen 7 7840HS / 16 GB RAM / RTX 4060 Laptop 8 GB (sm_89, Ada) / driver 592.82 (CUDA up to 13.1) / 135 GB free on D:

Legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked / worked around

---

## PHASE 0 — Safety net & environment  — COMPLETE (commit 45d3d1e)
- [x] 0.1 git init + .gitignore + baseline commit — commit `ad0f4d8` "Baseline before fixes"
- [x] 0.2 Verify GPU — `nvidia-smi`: RTX 4060 Laptop GPU, driver 592.82, CUDA up to 13.1, 8188 MiB VRAM
- [x] 0.3 Copied .venv BROKEN as predicted — `pyvenv.cfg` home = `C:\Users\harik\...`; running it gave `No Python at '"C:\Users\harik\...\python.exe'`. Deleted, recreated with **Python 3.11.9**
- [x] 0.4 Installed **torch 2.12.1+cu126 / torchvision 0.27.1+cu126** — STABLE index, no `--pre`. Verified `capability : (8, 9)`, fp32 4096^2 matmul + fp16 autocast both executed, no "no kernel image" error
- [x] 0.5 Deps: opencv-python 5.0.0.93, scipy 1.17.1, PyYAML, pytorch-wavelets 1.3.0, timm, tensorboard, tqdm, matplotlib, **PyWavelets 1.9.0** (undeclared transitive dep — smoke test failed `ModuleNotFoundError: No module named 'pywt'` until added)
- [x] 0.6 requirements.txt rewritten as UTF-8 (`file` reports `UTF-8 text`) with environment notes
- [x] 0.7 Verified-then-archived to `_archive/`: nested dup (11 scripts all md5-IDENTICAL + a 3rd broken 3.12 venv, 499 MB), models.zip (all 24 entries IDENTICAL to `models/`), `losses/` (0 files)
- [x] 0.8 `checkpoints/_baseline_before_fixes/{best,latest}.pt` — md5 verified (`dbc299e8fa73915ba5fb8ae703274c86`)
- [x] 0.9 `smoke_test_amp.py` -> "Smoke test with AMP passed!"; `smoke_test_sm120.py` -> "Smoke test passed! No CUDA errors" (fine — this GPU is sm_89, not sm_120). **GATE**: `tools/gate_check.py` ran 5 real training steps on real UIEB pairs: **loss 0.852788 -> 0.608016 (delta -0.2448)**, peak VRAM 7.19 GB @ bs=8
- [x] 0.10 Commit `45d3d1e`

## PHASE 1 — Fix core attention bug — COMPLETE (commit 3580b40)
- [x] 1.1 Real Q/K/V + out projections, multi-head split, in `models/attention/physics_attention.py`
- [x] 1.2 Physics bias upgraded: rank-1 scalar outer product -> per-head per-position projected bias `(B, H, 1, N)`
- [x] 1.3 `models/build.py::build_model()` is the only model constructor; all 8 entry points updated; class default `num_heads` 1 -> 4
- [x] 1.4 `L_frequency` (Haar-DWT sub-band L1) + `lambda_frequency: 0.15` in config
- [x] 1.5 **VERIFIED** — `tools/verify_attention.py`, all 4 checks PASS (see evidence below)
- [x] 1.6 Commit `3580b40`

### Phase 1 evidence (`python tools/verify_attention.py`)
```
CHECK 1 - convex-hull limitation (the actual bug)
  task: learn tokens -> tokens + 3.0 (a pure colour shift, outside the hull)
  OLD module learnable params :    66  (no Q/K/V projections at all)
  NEW module learnable params : 66309
  input  mean : -0.0069   target mean : +2.9931
  OLD output mean after fitting: -0.0069   final MSE:   9.00000   <-- 9.0 == 3.0^2, learned NOTHING
  NEW output mean after fitting: +2.9943   final MSE:   0.39800
CHECK 2 - output responds to the physics feature map : 146.76% relative change  PASS
CHECK 3 - num_heads changes the computation         : mean |diff| 0.256413    PASS
CHECK 4 - full model fwd+bwd, 2,729,450 params, all grads finite, peak VRAM 2.60 GB (bs=4)  PASS
```
The old attention module was mathematically incapable of a global colour shift — the
single most important operation in underwater colour correction. That is the bug.

### VRAM went DOWN despite adding parameters
Switching to `scaled_dot_product_attention` with a broadcastable `(B, H, 1, N)` physics
bias means the `(B, N, N)` score matrix is never materialised. Measured peak at bs=8:
**7.19 GB (old) -> 5.19 GB (new)**. bs=12 also fits (7.76 GB). Kept bs=8 to match the
baseline's training config so the comparison stays fair, and to leave GPU headroom.

## PHASE 2 — Eval pipeline correctness — COMPLETE (commit 3580b40)
- [x] 2.1 `data/dataset.py::get_splits()` — one seeded 90/10 split shared by train.py, validate.py, test.py. **Verified: 801 train / 89 val / 890 total.** validate.py & test.py gained `--split {val,train,full}`, defaulting to held-out `val`
- [x] 2.2 `training.save_every_epochs` now writes `checkpoints/epoch_N.pt` (was configured but never used); `early_stopping.metric` now honoured (was hardcoded to PSNR); dead `scheduler.total_epochs` key removed
- [ ] 2.3 Re-verify detection "grid of boxes" does not reproduce — deferred to Phase 3 (the new detector replaces that code path entirely; will verify on the fine-tuned model)
- [x] 2.4 Commit `3580b40`

## PHASE 3 — Real underwater object detection — COMPLETE
- [x] 3.1 Dataset: **RUOD** (the primary recommendation) — 14,000 real underwater images, 10 marine classes, fetched from the public HF repo `Mortallll/RUOD`, no auth needed
- [x] 3.2 Converted COCO -> YOLO: **9,800 train / 4,200 val, 51,932 + 22,968 boxes, 0 missing, 4 degenerate boxes dropped**. Labels visually verified by re-rendering onto images (`outputs/_phase3_check/label_sanity_check.png`)
- [x] 3.3 Fine-tuned **YOLO11n**, 20 epochs @ 640 px, batch 16, 2.4 GB VRAM (`logs/detection_train.log`)
- [x] 3.4 Wired into `models/object_detection.py` (`build_detector`) as the default; legacy COCO path kept as `--detector fasterrcnn`
- [x] 3.5 **mAP@0.5 = 0.8292**, mAP@0.5:0.95 = 0.5845, P = 0.8385, R = 0.7561 -> `results/detection_metrics.json`
- [x] 3.6 End-to-end panels for 3 held-out images -> `outputs/_phase3_check/`
- [x] 3.7 Commit

### Phase 3 evidence
Per-class AP@0.5: cuttlefish 0.965, turtle 0.965, diver 0.929, echinus 0.880, starfish 0.862,
jellyfish 0.787, holothurian 0.751, fish 0.746, scallop 0.714, corals 0.694.

Same enhanced frames, both detectors:
```
708_img_  YOLO: 2 boxes [corals, fish]        | legacy: 0 boxes (found nothing)
493_img_  YOLO: 2 boxes [cuttlefish, turtle]  | legacy: 3 boxes 'sea_bird' (raw COCO: bird)
497_img_  YOLO: 18 boxes [fish]               | legacy: raw COCO labels = bird, frisbee, spoon, toothbrush
```

### Phase 2.3 (deferred) — "grid of hundreds of boxes" re-verified ABSENT
```
max boxes on any sample: 18 (sane limit 60)
'grid of hundreds of boxes' bug reproduced: False
```
`tools/detection_check.py` now asserts this automatically, so a regression is caught rather
than eyeballed.

## PHASE 4 — Enhancement training run — COMPLETE (time-boxed)
- [x] 4.1 Launched as a background process -> `logs/train.log`
- [x] 4.2 Monitored throughout; two interventions logged (D-009 re-scope, D-012 crash + resume)
- [x] 4.3 Leak-free `validate.py` on the held-out 89: **PSNR 24.9558, SSIM 0.9261, UIQM 10.1351, UCIQE 0.3125**
- [x] 4.4 Six sample triptychs -> `outputs/_final_check/`; before/after panels -> `outputs/_phase1_check/`
- [x] 4.5 Commit

**Result stated plainly: the retrained model did NOT beat the baseline.**
| Model | Epochs | PSNR | SSIM |
|---|---|---|---|
| Baseline (pre-fix) | 115 | 25.114 dB | 0.9281 |
| Retrained (post-fix) | 50 | 24.956 dB | 0.9261 |

It was still improving at the epoch limit (last five validations 24.78 / 24.81 / 24.77 /
24.79 / 24.81). The comparison is not like-for-like on training budget — see FINAL_REPORT.md §4.1.
No CUDA OOM occurred at any point, so the halve-batch-size contingency was never needed.

## PHASE 5 — Final report — COMPLETE
- [x] 5.1 `FINAL_REPORT.md` written
- [x] 5.2 README drift fixed (nonexistent `infer.py --batch-size`, LR 1e-4 -> 2e-4, batch 4 -> 8, new config rows, detection usage, updated repo tree); `docs/architecture.md` Module 5 implementation notes + new Module 8 on the detection stage

---

## Decision log
(Every unilateral decision, with reasoning, appended here.)

### D-001 — .gitignore scope
Added `datasets/`, `*.pt/*.pth/*.onnx`, `_archive/`, `runs/` to the existing .gitignore.
Reason: repo stays lightweight; datasets (1.4 GB) and checkpoints (48 MB) are reproducible/backed up on pendrive.
Existing entries already covered `.venv/`, `checkpoints/`, `logs/`, `outputs/`, `__pycache__/`.
NOTE: `logs/` and `outputs/` being gitignored means the evidence artifacts I produce are NOT committed —
they live on disk for the user to inspect. FINAL_REPORT.md references them by path.

---

## Running notes

### D-002 — Python 3.11 instead of the original 3.12
Original venv was 3.12.10; only 3.11 / 3.13 / 3.14 exist on this machine. Chose **3.11**
for the widest wheel availability across torch, torchvision, ultralytics and opencv —
the safest choice for a 6-hour unattended run.

### D-003 — torch 2.12.1 + cu126 STABLE (the key environment change)
Driver 592.82 advertises CUDA up to 13.1, so cu126 / cu128 / cu130 would all run. Probed
all four indexes; picked **cu126** (newest stable torch line, most battle-tested) and
pinned torch to **2.12.1** — the same version the project's own requirements.txt targeted,
so only the CUDA build changed, not the framework version. Explicitly NOT nightly: this
GPU is sm_89 (Ada), fully covered by stable wheels. Evidence: `get_device_capability(0)`
returned `(8, 9)` and a real 4096x4096 CUDA matmul executed.

### D-004 — torchaudio omitted
The cu126 index tops out at torchaudio 2.11.0, which does not pair with torch 2.12.1
(precisely the mismatch the old requirements.txt had pinned). torchaudio is imported
nowhere in this project, so I dropped it rather than downgrade torch for it.

### D-005 — whole nested duplicate folder archived, including its dead venv
The nested folder also held a 499 MB copy of the broken 3.12 venv (same hardcoded
`C:\Users\harik\...` path, equally unusable). Moved the entire folder to `_archive/`
rather than deleting the venv separately: a rename is instant and fully reversible,
and 135 GB free means the space is irrelevant.

### D-006 — VRAM headroom is thin (drives the Phase 1 design)
Peak VRAM was **7.19 GB of 8.19 GB at batch_size=8** on the UNMODIFIED model. Root cause:
attention materialises a (B, N, N) score matrix with N = 64x64 = 4096 tokens
(8 x 4096 x 4096 fp16 = 268 MB per branch, plus autograd saves). A naive multi-head
rewrite would make this (B, H, N, N) = 4x worse and OOM immediately. See D-007.

### D-007 — attention rewritten around scaled_dot_product_attention (memory)
D-006 showed only 1 GB of VRAM headroom. A textbook multi-head rewrite materialises a
`(B, H, N, N)` score tensor — at N=4096, H=4, bs=8 that is ~1 GB per branch in fp16 plus
autograd saves, and would have OOMed immediately. Two choices avoided that:
  * the physics bias is per-key, shape `(B, H, 1, N)`, which broadcasts over queries.
    This is also the faithful reading of docs/math.md sec 5 ("P is the projected physics
    feature map" — a feature map has one value per spatial position, not N^2 values).
  * `F.scaled_dot_product_attention` then never materialises the N x N matrix.
Net: peak VRAM at bs=8 went DOWN from 7.19 GB to 5.19 GB while ADDING ~66k parameters
per attention module.

### D-008 — GPU is power-capped to 19 W of 77 W; training re-planned around it  [IMPORTANT]
Roughly 20 minutes into the main run, epochs went from ~55 s to ~146 s and stayed there
even after all background downloads finished. Diagnosis:

```
temperature.gpu  52 C          <- NOT thermal
clocks.sm        585 MHz       <- of 3105 MHz max (19%)
power.draw       19.46 W       <- of a 77 W board
clocks_event_reasons.active  0x4  = SW Power Cap
Win32_Battery BatteryStatus = 2 (on AC), charge 100%
Active power scheme: Balanced
```
So the machine is plugged in and cool, but Windows' Balanced power mode (or the OEM
embedded controller) holds the dGPU at ~25% of its clock under sustained load. The 2.65x
slowdown matches the 55 s -> 146 s change exactly.

**I did not change the power plan.** Altering system power settings on someone's machine
unattended is out of scope for this task, and it would persist after I finish. Instead I
re-planned the training to fit the compute actually available.

If the user wants the ~2.6x back, from an **Administrator** PowerShell:
```
powercfg /setactive SCHEME_MIN          # High performance
```
and set Settings > System > Power & battery > Power mode = "Best performance".
Re-running the same training at full clock should take ~55 s/epoch instead of ~146 s.

### D-009 — enhancement run re-scoped from 150 to 65 epochs
At 146 s/epoch, 150 epochs is 6.1 hours — more than the entire session budget, and it
would have left no time for the detection work (the advisor's actual ask). Options were
(a) stop at ~epoch 80 with the cosine LR schedule only half-finished, leaving the model
un-annealed and needlessly undertrained, or (b) resume with a shorter total so the
warmup+cosine schedule completes properly.

Chose (b): stopped at epoch 19 and resumed from `checkpoints/latest.pt` with
`--epochs 65`. Because the LR is computed from `global_step / total_steps`, resuming at
step 1900 against a 6500-step total puts it correctly at 24% of the schedule
(observed LR on resume: 1.74e-4), and it now anneals to `eta_min` by epoch 65.
No progress was discarded. Cost: a model trained 65 epochs rather than 150.
This is a compute-budget limitation, not a convergence result — stated as such in
FINAL_REPORT.md.

### D-010 — UCIQE metric was broken (audit said it was fine; it was not)  [AUDIT CORRECTION]
The prior audit listed the PSNR/SSIM/UIQM/UCIQE implementations under "Confirmed NOT
broken". Verifying anyway (as instructed) turned up two genuine bugs in
`metrics/uciqe.py`. The first leak-free `validate.py` run reported:

```
UCIQE : 17128886.2888        <- should be roughly 0.2 - 0.7
```

Root causes:
1. **`a`/`b` never re-centred.** OpenCV's 8-bit LAB stores a and b offset by +128, so a
   neutral grey pixel is (128, 128), not (0, 0). `sqrt(a^2 + b^2)` therefore scored a
   colourless pixel as maximally chromatic. Measured on a real UIEB reference image:
   mean chroma 166.1 uncentred vs 21.5 centred — a 7.7x inflation of the sigma_c term.
2. **Division by luminance with no guard.** `saturation = chroma / (L + 1e-10)`, and
   OpenCV's L hits exactly 0 on black pixels, giving saturation ~1.8e12. A fully black
   frame scored **4.66e11**. Any image with pure-black pixels poisoned the average.

Fix: normalise L to [0,1], re-centre a,b to [-0.5,0.5], and use the bounded saturation
form `chroma / sqrt(chroma^2 + L^2)`, which is in [0,1] by construction.

Verified after the fix:
```
reference image UCIQE: 0.2936     raw (degraded) image UCIQE: 0.2750
all-black: 0.0  (was 4.66e11)     all-white: 0.0     random noise: 0.3583
```
Reference scoring above raw is the expected direction. **Any previously reported UCIQE
number for this project is meaningless**; all UCIQE figures in FINAL_REPORT.md are
recomputed with the fixed metric for both the baseline and the retrained model.

NOTE on UIQM: it reports ~10.1-10.4, which is high versus the ~2-5 commonly quoted for
UIEB. I did **not** change it — unlike UCIQE it is not numerically broken (no blow-up, no
degenerate values), and the discrepancy looks like a coefficient/scaling convention
difference between UIQM variants. Flagged for the user rather than silently altered,
since changing it would break comparability with previously logged runs.

### D-011 — leakage confirmed exactly; num_heads mismatch measured as negligible
`tools/eval_baseline.py` re-scores the ORIGINAL pre-fix checkpoint (epoch 115) using the
pre-fix architecture from the `_archive/baseline_code` worktree, but with the corrected
split and the fixed UCIQE.

**Train/eval leakage — audit confirmed, essentially to the decimal:**
```
old leaky all-890 evaluation would report : PSNR 27.2414 dB   (logs/test.log said 27.24)
honest held-out figure (89 images)        : PSNR 25.1141 dB
inflation from leakage                    : +2.1274 dB
    held-out 89 : PSNR 25.1141   train 801 : PSNR 27.4778
```
The model scores 2.36 dB higher on images it trained on. That is the whole gap.

**num_heads mismatch — REAL bug, but measured impact ~0:**
```
baseline weights @ num_heads=1 (what every eval script used): PSNR 25.1141 dB
baseline weights @ num_heads=4 (what train.py actually used): PSNR 25.1151 dB
cost of the silent mismatch                                 : +0.0010 dB
```
The audit called this CRITICAL. The code defect was exactly as described, but on this
checkpoint it costs one thousandth of a dB, so it is NOT an explanation for any observed
quality problem. Reported honestly rather than claimed as a win.

The reason is instructive: in the OLD module `num_heads` only altered the softmax scale
divisor (`head_dim**0.5`), since Q=K=V and no head split ever happened. **After the Phase 1
fix, `num_heads` genuinely changes the computation** — `verify_attention.py` check 3
measures a 0.256 mean-absolute difference between 1 and 4 heads on identical weights. So
`models/build.py` is load-bearing *going forward* even though it recovered no accuracy
retroactively.

### Baseline reference numbers (all later comparisons use these)
Pre-fix checkpoint, epoch 115, held-out 89 images, fixed UCIQE:
| PSNR | SSIM | UIQM | UCIQE |
|---|---|---|---|
| 25.1141 dB | 0.9281 | 10.0009 | 0.3133 |

### D-012 — training crashed at epoch 26 (my fault); recovered from checkpoint
`logs/train_stdout2.log`:
```
[W CachingHostAllocator.cpp:26] Warning: Exception in pinned allocator free(), rethrowing
Unhandled exception caught in c10/util/AbortHandler.h
```
Host RAM exhaustion, not VRAM. I ran `tools/eval_baseline.py` (scoring 801+89+89 images)
and a native-resolution `infer.py` (1536x814) on CPU *while* training was running with
`pin_memory: true`. On a 16 GB machine that starved the pinned-memory allocator and killed
the training process. Entirely self-inflicted — a lesson about this machine's RAM budget,
not a defect in the training code.

Recovery cost nothing: `checkpoints/latest.pt` held epoch 25 and `best.pt` epoch 22, so I
resumed from epoch 25. This is exactly the failure the per-epoch checkpointing (and the
newly-wired `save_every_epochs` snapshots) exist for.

Rule adopted for the rest of the session: **no heavy CPU jobs while training runs.**

### D-013 — enhancement target cut again, 65 -> 50 epochs
After the crash the arithmetic no longer left room for the detection work, which is the
advisor's actual ask and had no results yet. Resumed from epoch 25 with `--epochs 50`
(25 more epochs, ~1.05 h) so the cosine schedule still anneals to `eta_min` at the new
horizon, then hand the GPU to detector fine-tuning. Sequencing detection *after* a
shortened enhancement run (rather than before) keeps a converged enhancement checkpoint
while still guaranteeing the detection deliverable lands.

---
---

# SESSION 2 — 2026-08-20 (evening) — Fair-Comparison Retrain + Autonomous Watchdog

**Session start:** ~21:30 local, unattended (bypass-permissions).
**Goal:** give the fixed architecture a genuinely fair, full-length training run (the
50-epoch result from session 1 was time-boxed, not converged), with automatic crash
recovery so nothing stops it finishing.

Legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked / worked around

## PHASE 0 — Verify state, fix GPU throttle, back up
- [x] 0.1 `git log` confirms 12 commits, HEAD = `e626cac`
- [!] 0.2 **`PROGRESS.md` was missing from disk** — `git status` showed ` D PROGRESS.md`. Restored intact from HEAD (304 lines) with `git checkout -- PROGRESS.md`. Only that one file was affected; everything else was clean. Cause unknown (deleted between sessions). Session 1's history is preserved and tonight is appended below it, as instructed.
- [x] 0.3 Backed up the 50-epoch post-fix run to `checkpoints/_50epoch_postfix_backup/` — md5 verified (`best.pt` = `49444aace20caca676a04f0d87129c18`, `latest.pt` = `7ed24e40b1beaef47967ffcf037fbe1a`)
- [x] 0.4 Archived session 1's log to `logs/train_50epoch_run.log` (721 lines) so tonight starts a clean `logs/train.log`
- [x] 0.5 **GPU power throttle FIXED** — see S2-D-001 below
- [x] 0.6 Env sanity: `torch 2.12.1+cu126  True  NVIDIA GeForce RTX 4060 Laptop GPU`
- [x] 0.7 Commit

### S2-D-001 — GPU throttle fix: what worked, and the measurement that proves it
Session 1 declined to change system power settings unattended; tonight's brief explicitly
authorised it, so I did.

`powercfg /setactive SCHEME_MIN` could not work as written: **only the Balanced scheme
existed** on this machine (Windows 11 hides the legacy High Performance scheme), and the
shell is **not elevated**. What worked, without admin:

```
powercfg /duplicatescheme 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c   -> created 5670f263-...  (High performance)
powercfg /setactive       5670f263-ce63-477a-91c0-522d01182f6d   -> exit 0
powercfg /getactivescheme -> Power Scheme GUID: 5670f263-...  (High performance)
```
(`powercfg /overlaysetactive OVERLAY_SCHEME_MAX` returned exit 0 but this build does not
support `/getactiveoverlayscheme`, so its effect could not be confirmed independently.)

**Measured, not assumed** — `tools/gpu_power_probe.py` runs 150 s of *real* PFGT-UIE
training steps (the cap only engaged under sustained load last night, so a short burst
would have proved nothing) and samples nvidia-smi throughout:

| | Session 1 (Balanced) | Tonight (High performance) | Change |
|---|---|---|---|
| Power draw | 19.46 W | **79.36 W** | **4.1x** |
| SM clock | 585 MHz (18.8% of max) | **2514 MHz (81.0% of max)** | **4.3x** |
| Measured epoch time | ~146 s | **~50 s** | **2.9x faster** |
| Temperature | 52 C | 47 -> 77 C over 150 s | runs hot, as expected |

**Important subtlety — `SwPowerCap` is still reported, and that is correct.** The flag was
active in 13/14 steady-state samples even after the fix. It does not mean "throttled"; it
means "currently limited by the power budget", which is the normal state of any GPU running
flat out at its board TDP. At 79 W the card is sitting at ~103% of its 77 W limit, i.e. at
full power — versus 19 W (25%) last night. **The flag alone is not the signal; the magnitude
is.** My probe's first verdict heuristic got this wrong (it required the flag to be absent)
and reported "STILL THROTTLED" against its own data showing a 4x improvement; I corrected
the logic in `tools/gpu_power_probe.py` rather than trusting the wrong label.

Budget consequence: **150 epochs x ~57 s (incl. validation) ~= 2.4 hours**, comfortably
inside tonight's window. Session 1's throttled estimate for the same run was 6.1 hours.

Watch item for the watchdog: the die reached 77 C within 150 s. If sustained training
drives it to the thermal limit, `SwThermalSlowdown` / `HwThermalSlowdown` would appear and
clocks would fall. The probe now warns on those bits specifically.

## PHASE 1 — Launch the fair-comparison run
- [x] 1.1 Launched fresh (no `--resume`, no `--epochs` override) so early stopping decides the real endpoint, exactly as the 115-epoch baseline was produced:
      `python train.py --config configs/train.yaml --num-workers 2` -> `logs/train.log`
- [x] 1.2 Confirmed real training within the first minute (not silence, not an instant crash)
- [x] 1.3 Commit

Startup evidence:
```
Config: epochs=150  bs=8  lr=2.00e-04  amp=True  workers=2  grad_clip=1.0
Built PFGTUIEModel(embed_dim=128, num_heads=4)
Model parameters: 2,729,450
Dataset split: 801 train / 89 validation images (shared get_splits, seed=42)
epoch=1/150  step=10  loss=0.851062 ...
=== Epoch 7/150 finished | avg_loss=0.180495 | time=49.7s ===
  Validation -> val_loss=0.176214  PSNR=23.0208 dB  SSIM=0.9030
```
Two useful confirmations there: **49.7 s/epoch** (vs 146 s last night — the throttle fix
holding under real training load, not just in the probe), and `epoch=1 step=10
loss=0.851062` reproduces session 1's `0.851053` to 5 decimal places, confirming seed-42
determinism across the whole pipeline.

Old periodic snapshots (`epoch_10..50.pt` from the 50-epoch run) were moved into
`checkpoints/_50epoch_postfix_backup/` first, so tonight's `epoch_N.pt` files cannot be
confused with last night's.

## PHASE 2 — Autonomous watchdog
- [x] 2.1 Built `tools/watchdog.py` and left it running for the night
- [x] 2.2 Verified all three of its decision paths BEFORE trusting it (see S2-D-002)

### S2-D-002 — the watchdog, and the two failure modes I found while testing it
The brief asked for a check-and-recover cycle every 20-30 min. A polling loop that only I
drive has a worst case of ~30 min of dead GPU time per incident, and cannot act at all
between turns, so I wrote a resident watchdog that polls every 60 s and restarts
automatically. I still check in periodically on top of it.

It is deliberately trivial in cost — it reads a log file and scans the process table, and
never loads a model — so it cannot repeat session 1's crash (D-012), where a CPU-heavy
evaluation run alongside training exhausted host RAM.

**Testing it against the live run surfaced two bugs that would have made it useless:**

1. **False "alive" from launcher wrappers.** A `nohup -> .venv shim -> real python` chain
   produces *three* processes whose command line is `train.py --config ...`:
   ```
   [(7320, '2253MB'), (23196, '6MB'), (25400, '4MB')]
   ```
   Only pid 7320 is the real trainer. If it died and a 6 MB wrapper lingered, a naive
   presence check would report "healthy" and the watchdog would never restart — the exact
   failure a watchdog exists to prevent. Fixed by requiring >=200 MB RSS to count as the
   real trainer.

2. **No stall detection at all.** The brief lists "a crash, an OOM, a stall". Process
   presence catches the first two but not a hang (deadlocked dataloader, wedged CUDA
   call) where the process stays resident and silent. Added: if `logs/train.log` goes
   unmodified for 900 s while the process lives, kill the whole process tree and resume
   from `latest.pt`. Epochs take ~50 s and a step line lands every ~5 s, so 15 minutes of
   silence is unambiguous.

**All three decision paths verified against reality rather than assumed:**
```
training_running()   -> 7320          (picks the 2253MB trainer, ignores the 6MB/4MB shims)
log_says_done()      -> None          on the live log
log_says_done()      -> 'Training complete.'   on the archived finished 50-epoch log
halve_batch_size()   -> (8, 4) then (4, 2)     then restored to 8, config identical to HEAD
```
That last one matters: an OOM handler whose regex silently misses would leave the run
OOM-looping all night. Tested it explicitly, then `git checkout -- configs/train.yaml`.

Restart semantics: resumes with `--resume checkpoints/latest.pt` and **no** `--epochs`
override, so `train.py` keeps the 150-epoch total from the config and its LR stays on the
original warmup+cosine curve (`global_step / total_steps`) rather than restarting the
schedule — the mistake session 1's D-009 warned about. Safety valve: 25 restarts max, with
escalating backoff if a restart dies inside its grace period.

### S2-D-003 — a hot-swap left TWO watchdogs running; caught by a process census
While adding a heartbeat I restarted the watchdog. The `terminate()` did not take on the
old instance, and a census found two live:
```
real trainers : [(7320, '2033MB')]
real watchdogs: [(10700, '21MB'), (19736, '21MB')]
WARNING: expected exactly 1 watchdog
```
Two watchdogs is **worse than none**: on a crash both would restart training, producing two
concurrent runs fighting over the same GPU and the same `checkpoints/latest.pt`. Killed the
older (10700) by create-time and confirmed `CLEAN: 1 trainer, 1 watchdog`.

Two things came out of it:
* `tools/_proccheck.py` — a read-only census that exits non-zero unless exactly one real
  trainer and one real watchdog are alive. Used at every check-in from here on.
* A **singleton guard** in `tools/watchdog.py`: a second instance now refuses to start.
  Verified: `ABORT: another watchdog is already running (pids [19736]).`
  Note the currently-running instance (19736) was started *before* the guard was added, so
  the guard protects future launches, not this one — the census is what protects tonight.

Also fixed a bad monitoring habit: `ps -W` on this machine prints only the executable path,
**not the arguments**, so `ps -W | grep watchdog.py` never matches and any wait-loop built
on it exits instantly (my first completion waiter did exactly that, firing at epoch 19 of
150). All process checks now go through psutil, which sees full command lines.

## PHASE 3 — Fair comparison + evidence
- [x] 3.1 Leak-free `validate.py` on the held-out 89 (fp32, fixed UCIQE)
- [x] 3.2 Three-way comparison table
- [x] 3.3 Sample panels -> `outputs/_final_check_150ep/` (last night's `_final_check/` untouched)
- [x] 3.4 Commit

### The run finished on its own terms
```
Early stopping triggered: PSNR did not improve for 20 validation rounds.
Training complete. Best PSNR: 24.7587
Best checkpoint: checkpoints/best.pt   (epoch 81, step 8100)
```
101 epochs run, best at epoch 81, then 20 validation rounds without improvement. **It
converged — it was not cut short.** That matters more than any single number here, because
it removes last night's excuse.

### Headline: the three-way comparison (held-out 89 images, fp32, fixed UCIQE)

| Model | Epochs | PSNR | SSIM | UIQM | UCIQE |
|---|---|---|---|---|---|
| Pre-fix baseline | 115 | **25.114 dB** | **0.9281** | 10.001 | 0.3133 |
| Post-fix, time-boxed (session 1) | 50 | 24.956 dB | 0.9261 | **10.135** | 0.3125 |
| **Post-fix, converged (tonight)** | **101 (best @ 81)** | 24.902 dB | 0.9267 | 9.963 | **0.3144** |

**Verdict: the fix did not pay off on PSNR/SSIM, and the "it just needs more epochs"
explanation is now dead.** Given a full, fair, converged budget on an un-throttled GPU, the
fixed architecture lands **0.212 dB below** the pre-fix baseline — and fractionally below
even its own 50-epoch version. Session 1 predicted the gap would close with training; it did
not. That prediction is falsified, and I am recording it as such.

SSIM is effectively a tie (−0.0014). UCIQE is marginally the best of the three (+0.0011).
Neither changes the conclusion.

### Sample-level detail — `outputs/_final_check_150ep/`
Four-panel comparisons, `raw | pre-fix 115ep | post-fix converged | reference`:
```
image             PSNR ref|OLD  PSNR ref|NEW
708_img_.png             26.87         21.38
15094.png                23.83         27.97
356_img_.png             27.44         28.20
372_img_.png             18.66         18.13
290_img_.png             24.53         22.75
172_img_.png             28.84         25.81
MEAN                     25.03         24.04
```
Note this is **not** a uniform regression — the new model clearly wins on `15094` (+4.1 dB)
and `356` (+0.8 dB) and clearly loses on `708` (−5.5 dB) and `172` (−3.0 dB). The aggregate
gap is an average over genuinely mixed per-image outcomes, not a flat degradation.

### S2-D-004 — an important confound in this comparison, stated up front
The two post-fix rows differ from the baseline in **two** ways, not one:
1. the rebuilt physics-guided attention (real Q/K/V, multi-head, per-head physics bias), and
2. the added `L_frequency` term (Haar-DWT sub-band L1, `lambda_frequency: 0.15`), which
   session 1 introduced because `docs/math.md` §8 specifies it but it was never implemented.

So this table compares **pre-fix codebase vs post-fix codebase**, which is exactly the
question tonight's brief asked. It does **not** isolate the attention change on its own. If
the goal becomes "what did the attention fix specifically cost or buy", the clean control is
one more run at `lambda_frequency: 0.0` with everything else identical. Flagged rather than
glossed over, because the headline number would otherwise be over-attributed to the
attention rewrite alone.

### What this does and does not say
It does **not** invalidate the session 1 finding. The old attention module was provably
incapable of a global colour shift (`tools/verify_attention.py`: MSE 9.000 = 3.0², output
mean unmoved), and that remains true. What tonight establishes is narrower and more useful:
**making the project's headline novelty actually function does not, by itself, improve PSNR
on UIEB.** The enhancement network has several other learnable colour-remapping paths
(physics encoder convs, 1x1 band projections, fusion block, refinement head) that were
evidently already carrying that work. The fix is a correctness and honesty issue — the
paper's central claim now describes what the code does — not a metrics win.

## PHASE 4 (BONUS / EXPLORATORY) — does enhancement help downstream detection?
Chose option A from the brief (the detection ablation), not the TransformerBlock stacking.
Marked exploratory and kept separate from the Phase 3 headline result.

The experiment ran in three parts, because the first result was confounded and the second
test of that confound failed.

### Part 1 — the deployed pipeline (full 4,200-image val set)
`infer_detection.py` runs `raw -> enhance -> detect`, but the detector was fine-tuned on
**raw** RUOD. So: feed the same detector raw frames vs PFGT-UIE-enhanced frames.

```
raw      : mAP50 0.8292   mAP50-95 0.5845
enhanced : mAP50 0.7906   mAP50-95 0.5470
delta    : mAP50 -0.0386  mAP50-95 -0.0375
```
Harness self-check: the raw arm reproduces the detector's own training-time
`mAP50 = 0.8292` exactly, so the evaluation path is wired correctly.

Per-class AP50 delta was strikingly uneven — small benthic classes collapsed while large
distinctive ones barely moved:
```
jellyfish +0.0009  cuttlefish -0.0031  turtle -0.0059  fish -0.0081  diver -0.0089
corals -0.0158  holothurian -0.0777  echinus -0.0853  scallop -0.0873  starfish -0.0945
```

### Part 2 — my texture hypothesis, tested and FALSIFIED
The obvious reading was that enhancement smooths away the fine texture small seafloor
objects depend on. I measured it over 300 paired frames instead of asserting it:

| metric | raw | enhanced | change |
|---|---|---|---|
| Laplacian variance | 281.6 | 332.5 | **+18.1%** |
| high-pass energy | 74.17 | 103.24 | **+39.2%** |
| global contrast (std) | 37.95 | 53.05 | **+39.8%** |

Enhanced frames are **sharper and higher-contrast**, not blurrier (only 30% of images lost
Laplacian variance). The hypothesis was wrong, and the remaining explanation was domain
shift — which made the matched-domain test decisive rather than optional.

### Part 3 — matched-domain test (equal budget, only the domain differs)
Fine-tuned YOLO11n twice from scratch: 3,000 train frames, 20 epochs, same seed, batch and
imgsz — one arm on raw, one on enhanced, each evaluated on its **own** domain.

```
raw arm      : mAP50 0.5025   mAP50-95 0.2916
enhanced arm : mAP50 0.4845   mAP50-95 0.2778
delta        : mAP50 -0.0180  mAP50-95 -0.0138
```

### Finding
**PFGT-UIE enhancement does not help underwater detection on RUOD — it hurts, in both
conditions.** Decomposing the deployed pipeline's 3.9-point loss:

| Component | mAP50 |
|---|---|
| Deployed pipeline loss (raw-trained detector, enhanced input) | −0.0386 |
| ...of which is train/test **domain shift** (recoverable by retraining) | ≈ −0.0206 |
| ...of which is a **genuine residual cost** even when matched | **−0.0180** |

So roughly half the damage is a wiring problem the user can fix today by retraining the
detector on enhanced frames — and the other half is intrinsic: enhancement makes frames
look better to a human, and measurably raises contrast and high-frequency energy, but adds
no information the detector can use, while perturbing cues it had calibrated on.

**Immediately actionable:** the repo's current `enhance -> detect` default costs ~3.9 mAP50
points. Detect on raw frames and use enhancement for human review, or retrain the detector
on enhanced frames to recover about half of it.

### Caveat on Part 3
The matched arms used 3,000 images / 20 epochs, so both sit near mAP50 ≈ 0.50 rather than
the full detector's 0.829. The *comparison* is controlled (identical budgets, one variable),
but the conclusion is demonstrated at a lower operating point and may not transfer exactly
to the full-data regime. Repeating it at 9,800 images would settle that; it was not run
tonight to keep the report inside the session budget.

## PHASE 5 — Report
- [x] 5.1 `FINAL_REPORT_150EPOCH.md` written (session 1's `FINAL_REPORT.md` untouched, linked for prior context)

### Session 2 summary
| | |
|---|---|
| GPU throttle | FIXED — 19.46 W -> 79.36 W, 146 s/epoch -> 50 s/epoch (2.9x) |
| Training | 101 epochs, clean early stop at a converged optimum, best @ epoch 81 |
| Watchdog | **0 restarts, 0 incidents** — never needed to intervene |
| Fair comparison | post-fix **24.902 dB** vs pre-fix baseline **25.114 dB** -> the fix does NOT pay off |
| Bonus ablation | enhancement **hurts** detection: -0.0386 mAP50 deployed, -0.0180 matched-domain |
| Hypotheses falsified tonight | "more epochs will close the gap"; "enhancement blurs away texture" |

Machine left running and idle, as instructed — nothing shut down or slept.

---
---

# SESSION 3 — 2026-08-21 (evening) — Architecture Design Review + One Informed Run

**Premise:** this code was originally assembled from pasted LLM output, not engineered
module-by-module. Sessions 1-2 found the one defect with a visible symptom (attention had no
Q/K/V) and then ran empirical experiments around whatever else was there. This session
reviews the *rest* of the architecture on design merit before spending more GPU time.

## PHASE 0 — Fix the detection default (free win)
- [x] 0.1 `infer_detection.py` now detects on **raw** frames by default
- [x] 0.2 Verified with `tools/verify_detection_default.py` — both checks PASS
- [x] 0.3 Commit

Session 2 measured that the deployed `enhance -> detect` path costs 3.9 mAP@0.5
(0.8292 -> 0.7906) because the detector was fine-tuned on raw RUOD frames. The script was
paying that on every call. The fix separates the two consumers, which want different images:

* `--detect-on raw` (**new default**) — the detector gets the distribution it was trained on.
* `--annotate-on enhanced` (default) — the human still gets the readable, colour-corrected
  picture to look at.
* `--no-enhance` — skip enhancement entirely (fastest path).

Evidence (`tools/verify_detection_default.py`):
```
images tested                              : 12
raw vs enhanced detections genuinely differ: 12/12     <- the two paths are not equivalent
DEFAULT path matches RAW detections        : 12/12
DEFAULT path matches ENHANCED detections   : 0/12
--detect-on enhanced matches ENHANCED      : 12/12     <- old behaviour still reachable
detector on RAW RUOD val : mAP@0.5 = 0.8292            <- vs 0.7906 enhanced
```
The first line matters: it rules out the possibility that the check passes trivially because
both paths happen to produce identical detections.

## PHASE 1 — Design review of the remaining modules

Method: I read every module myself, then ran an independent parallel review (6 module critics
+ literature + completeness critic) to avoid anchoring on my own reading, and **measured**
every load-bearing claim rather than asserting it. Two of my own hypotheses were falsified by
those measurements — recorded below, because ruling things out is part of the result.

### Summary table

| Module | Verdict | Headline |
|---|---|---|
| `physics_encoder.py` | **actually-wrong** | No physics in it. Structurally cannot compute the global statistics that matter. |
| `model.py` (capacity split) | **actually-wrong** | 65% of parameters serve 2.7% of the error. |
| `transformer_block.py` | **weak-but-not-broken** | Exactly permutation-equivariant — no notion of position or locality. |
| `loss.py` (`L_frequency`) | **weak-but-not-broken** | 99.97% correlated with L1. It is a duplicate term. |
| `fusion.py` | see below | Concat + 1x1 conv, unconditioned on the physics signal. |
| `refinement.py` | see below | 4.8% of parameters for the stage that must fix a colour error. |
| InstanceNorm erasing the cast | **fine** (falsified) | Tested. It does not. |
| P reaching both branches | **fine** | README's claim holds. |

---

### F1 — Capacity is allocated backwards  [HIGH · strongest evidence in this review]

`tools/_errordecomp.py` decomposes the trained model's remaining held-out error into wavelet
bands and compares against where the parameters actually are:

```
band                          error energy  % of error
LL (colour/illumination)          0.022002       97.3%
LH (horizontal detail)            0.000240        1.1%
HL (vertical detail)              0.000298        1.3%
HH (diagonal detail)              0.000072        0.3%

low_freq_transformer  :   198,533  ( 7.3%)
high_freq_transformer : 1,774,725  (65.0%)   ratio high:low = 8.9x
```

**97.3% of what is still wrong with the output is low-frequency colour error, and the branch
responsible for it holds one ninth of the parameters of the branch responsible for the other
2.7%.** The independent reviewer reached the same conclusion from a different direction: the
high branch's FFN alone is 1,181,568 params = **43.3% of the entire model**, and instrumenting
the trained checkpoint on real UIEB images it contributes only **8.9%** of its own branch's
residual magnitude (the low branch's FFN contributes 27.1%).

Root cause is mechanical, not a considered choice: `model.py:62` builds the high branch with
`embed_dim = embed_dim * 3` (=384, because LH/HL/HH are concatenated), and
`transformer_block.py:28` defaults `mlp_ratio=4`, so the FFN width triples with it and the
parameter count grows with the square. Nobody chose this ratio; it fell out of a default.

### F2 — The transformer block has no spatial structure at all  [HIGH]

`tools/_permcheck.py` — permute the tokens and the physics grid by the same permutation:

```
max |block(P(x), P(pf)) - P(block(x, pf))| = 5.960e-07
output scale                               = 4.390
relative                                   = 1.358e-07   -> permutation-equivariant
adjacent-token swap diff                   = 3.576e-07   -> no locality either
positional-encoding references in models/ + train.py: 0
```

The block is **exactly permutation-equivariant** to float32 precision. LayerNorm is per-token,
the FFN is two `nn.Linear` so it is per-token, attention is set-based, and the physics bias has
shape `(B, heads, 1, N)` — key-side only, so it tells token *i* nothing about *i*.

Why this matters specifically here: `docs/architecture.md` Module 4 assigns the high-frequency
branch "Texture restoration, Edge enhancement, Fine detail recovery". An edge *is* a spatial
arrangement. A permutation-equivariant operator cannot represent one. The branch is being asked
to do a job its own structure forbids.

### F3 — The "physics encoder" contains no physics  [HIGH]

`models/physics_encoder.py` is `Conv3x3(3→64) → GELU → 2×ResBlock → Conv3x3(64→64)`. That is a
generic CNN. Its only input is the same RGB the wavelet branch already receives, so it can only
re-derive what the rest of the network could compute itself. Its output is the `P` in
`Softmax(QK'/√d + λP)V` and is also fed to fusion — so the entire "physics-guided" premise rests
on it.

Two concrete problems, both measured:

**(a) It structurally cannot see the statistics that matter.** Every layer is a 3×3 `Conv2d`;
there is no `AdaptiveAvgPool2d` and no `Linear` anywhere in the module. Image-wide per-channel
means are therefore not representable at any depth.

**(b) Those are exactly the statistics that predict the answer.** `tools/_physicsprobe.py` takes
8 cheap closed-form priors (dark channel, bright channel, local std, per-channel means, R/G and
R/B ratios) and regresses them against the per-channel gain the UIEB reference actually applies:

```
gain_R: R^2 = 0.244     gain_G: R^2 = 0.494     gain_B: R^2 = 0.624
strongest single correlations: mu_G vs gain_G -0.619, mu_B vs gain_B -0.602,
                               bcp vs gain_G  -0.573, R/B  vs gain_B +0.464
```

Roughly a quarter to two thirds of the colour correction is **linearly predictable from eight
numbers** the module cannot compute.

### F4 — `L_frequency` is a duplicate of L1  [MEDIUM · resolves session 1's confound]

`tools/_lossdiag.py`, on the trained model over the held-out set:

```
term                     raw   weight   weighted  % of total
l1_loss              0.05256     1.00    0.05256       36.1%
ssim_loss            0.07451     0.50    0.03726       25.6%
perceptual_loss      0.50827     0.10    0.05083       35.0%
frequency_loss       0.03186     0.15    0.00478        3.3%

corr(pixel L1, LL band L1)      = +0.9997     LL   mean 0.10307
corr(pixel L1, LH band L1)      = +0.3851     LH   mean 0.00885
corr(pixel L1, HL band L1)      = +0.3761     HL   mean 0.00959
corr(pixel L1, HH band L1)      = +0.2624     HH   mean 0.00498
corr(pixel L1, mean-of-4-bands) = +0.9914
```

The Haar LL band is a local average, so its L1 is **99.97% correlated with pixel L1** — and
because LL's magnitude is 10-20× the detail bands', it dominates the 4-band mean. `L_frequency`
as implemented is therefore, to 99% accuracy, **a second copy of L1 at weight 0.15**,
contributing 3.3% of the objective. It cannot be "pulling its weight" when it is collinear with
a term already at weight 1.0.

The genuinely non-redundant signal is in LH/HL/HH (correlations 0.26-0.39) — but averaging them
with LL drowns it.

### Checked and found FINE — worth recording, it rules things out

* **InstanceNorm2d does NOT erase the global colour cast.** I suspected it did (it normalises
  each channel of each sample to zero mean/unit variance, and a colour cast is a per-channel
  shift). `tools/_normprobe.py` applies a synthetic cast and measures the relative change in
  each representation: features change **more** after the norm, not less (39.8% vs 27.9% for a
  strong cast). The 1×1 conv mixes channels before the norm, so a per-channel input scale does
  not map to a per-channel normalised shift. **Hypothesis falsified.**
* **Physics guidance really does reach both branches.** README's claim holds — `model.py:94`
  passes `physics_features` to the low branch and `:108` to the high branch.
* **`transformer_block.py`'s pre-LN wiring is textbook-correct.** Both residuals branch off the
  un-normalised stream; `norm1` is correctly shared across q/k/v; `model.py:54-55` supplies the
  final LayerNorm the pre-LN convention requires.
* **`fusion_norm` being reused at `model.py:101` and `:138` is harmless.** `nn.InstanceNorm2d`
  defaults to `affine=False, track_running_stats=False`, so it holds no parameters and no state
  — it is a pure function, and reuse is a cosmetic smell rather than a bug.

### F5 — THE FINDING: the model has no pathway to emit a per-image colour correction  [HIGHEST]

Two independent reviewers converged on this from different modules, and I verified the
headline number myself (`tools/_oracle_dc.py`, held-out 89, converged model at 24.9015 dB):

```
condition                                        PSNR     SSIM    delta
model as-is                                   24.9015   0.9267    0.000
+ oracle per-image per-channel OFFSET         28.1038   0.9349   +3.202
+ oracle per-image per-channel AFFINE         30.3572   0.9485   +5.456

fraction of remaining error energy that is pure per-image per-channel DC: 43.1%
for scale, the gap to the pre-fix baseline is 25.114 - 24.902 =           0.212 dB
```

**A single per-image, per-channel constant is worth +3.20 dB — fifteen times the gap this
project has spent two sessions trying to close.** And nothing in the network could produce
one:

* `model.py:160-162` applies `InstanceNorm2d(affine=False)` to the refinement head's input,
  setting its per-image per-channel mean to **exactly zero** (measured GAP magnitude 7.0e-09).
* `model.py:138` does the same to the fusion output one line after it is computed, discarding
  53.8% of that output's energy.
* The refinement head's measured receptive field is **9x9**, with no global-pooling path, and
  every convolution in it was `bias=False`.
* The physics encoder was all 3x3 convolutions (F3), so it could not supply the statistic either.

So the last stage capable of applying a colour correction was handed a tensor with that exact
information deleted, and could only see through a 9x9 window. This is not a tuning problem;
it is a missing pathway. It also explains F1: "97.3% of the error is in the LL band" and
"43% of the error is a per-image DC term" are the same phenomenon seen at two resolutions.

My earlier `_normprobe.py` result is not contradicted — I had tested the *wrong* normalisation
(`low_projection_norm`, early in the encoder, which indeed does not erase the cast). The two
that matter sit immediately before the decoder.

---

## PHASE 2 — Fixes applied, each verified before training

All six verification checks pass (`tools/verify_session3_fixes.py`). Full output in the commit.

| # | Fix | Targets | Verified by |
|---|---|---|---|
| 1 | `GlobalColorCorrection` — per-image per-channel affine predicted from a physics context vector, applied to the decoded image, **identity at init** | F5 (+3.20 dB headroom) | checks 3 + 4 |
| 2 | Physics encoder rebuilt: 8 closed-form priors as input + a global-pooling context branch | F3 (no physics, no global stats) | checks 1 + 2 |
| 3 | `high_mlp_ratio` 4 -> 2 | F1 (65% of params serving 2.7% of error) | param count |
| 4 | `lambda_frequency` 0.15 -> **0.0** | F4 (99.97% duplicate of L1) | measured correlation |
| 5 | BatchNorm -> GroupNorm in refinement + fusion; final conv gains a bias | measured 0.42 dB train/eval gap at batch 8 | check 5 |
| 6 | Fusion residual block gains a second conv (branch previously ended in GELU, so it could raise a feature without bound but lower it by at most ~0.17) | asymmetric residual | compiles + forward |

**The model got SMALLER: 2,729,450 -> 2,308,723 parameters (-420,727, -15.4%).** That is
deliberate. If the retrained model improves, it cannot be attributed to added capacity.

### The decisive verification (check 4)
Freeze the entire network, fit **only** the 4,550 global-correction parameters against the
oracle per-image offset, and see how much of the headroom a physics-context-predicted affine
can actually express:

```
base (no correction)          : 11.2892 dB
oracle per-image offset       : 12.7508 dB   (headroom +1.462)
learned from physics context  : 12.6289 dB   (recovered +1.340)
fraction of headroom captured : 91.7%
```

**91.7%.** The pathway is not merely present, it is expressive enough to deliver almost all of
what the oracle offers — and it does so from the physics context alone, which is the whole
premise of a "physics-guided" model. (This is measured on an *untrained* network, so it
demonstrates the pathway's capacity, not the final model's quality.)

### Other verification highlights
```
CHECK 1  R/G prior falls monotonically with cast severity: 0.7250 -> 0.5800 -> 0.4350 -> 0.2900
CHECK 2  encoder now contains Linear/global-pool layers: True (old: False)
         context change 6.28% mild cast -> 17.95% severe cast
CHECK 3  global correction at init: max|out-in| = 0.000e+00, gain exactly 1.0, shift exactly 0.0
CHECK 5  refinement head max|train(x)-eval(x)| = 0.000e+00  (BatchNorm gave a nonzero gap)
CHECK 6  gradients reach context_mlp, se, the 11-channel stem, the predictor and the new bias
```

### S3-D-001 — `lambda_frequency` set to 0, not merely reduced
The brief allowed either 0 or a value below 0.15. The measurement (F4) makes it 0: the term is
99.97% correlated with pixel L1 on the LL band, which dominates the 4-band mean by 10-20x, so
it is a duplicate of a term already at weight 1.0. A duplicate cannot "pull its weight" at any
non-zero coefficient — reducing it would just shrink a redundancy rather than remove it.

This also resolves session 1's confound: the post-fix runs can no longer be explained by an
untested extra loss term.

**A genuinely non-redundant frequency loss is still available and worth trying later:** apply
it to LH/HL/HH **only** (correlations with pixel L1 of 0.26-0.39, i.e. real independent
signal) and drop LL, whose consistency L1 already enforces. I did not do that here because it
would add a new variable to a run that already bundles six changes.

## PHASE 3 — One informed training run
- [x] 3.1 Backed up session-2 checkpoints to `checkpoints/_session3_backup/` (md5 verified)
- [x] 3.2 GPU verified at full clock at launch (78.20 W / 2535 MHz / 53.7 s per epoch)
- [x] 3.3 Launched fresh, watchdog running (reused, not rebuilt); early stopping decided the end
- [x] 3.4 Leak-free `validate.py` on the held-out 89

```
Early stopping triggered: PSNR did not improve for 20 validation rounds.
Training complete. Best PSNR: 25.2139     (best checkpoint: epoch 76, 96 epochs run)
Watchdog: 0 restarts, 0 incidents.
Held-out fp32: PSNR 25.3644  SSIM 0.9289  UIQM 10.1158  UCIQE 0.3221
```

### S3-D-002 — GPU re-throttled mid-run; I did not kill the user's application
Epoch time went 53.7 s -> ~146 s around epoch 52. Diagnosis: the power plan had **not** reverted
(still High performance) and the card was cool (54 C), but it re-entered `SwPowerCap` at ~19 W
of 77 W, and a second GPU client had appeared — `ChatGPT.exe`'s on-device model service.
Re-applying `powercfg /overlaysetactive OVERLAY_SCHEME_MAX` gave partial recovery
(390 -> 690 MHz). I left the user's application alone: killing a running app of theirs is not a
call I should make unattended. The run completed correctly, just slower.

### S3-D-003 — a false "FINISHED" signal, caught
My first completion waiter reported finished at epoch 13 of 150. Cause: `tools/_waitdone.py`
had been deleted in session 2's cleanup commit, so it failed instantly with "can't open file",
and the unguarded `echo "=== FINISHED ==="` after it printed anyway. Recreated the waiter with
a `test -f` guard and a stricter condition (process gone **AND** a completion marker in the
log, so a crash can never read as a finish). Also archived session 2's stale
`watchdog_incidents.json`, which would otherwise have made this session's clean run appear to
contain a prior "finished" event.

## PHASE 4 — Comparison + report
- [x] 4.1 `FINAL_REPORT_SESSION3.md`
- [x] 4.2 Comparison panels -> `outputs/_final_check_session3/` (earlier sessions' evidence untouched)

| Model | Epochs | Params | PSNR | SSIM |
|---|---|---|---|---|
| Pre-fix baseline | 115 | 2,729,450 | 25.114 | 0.9281 |
| Post-fix, time-boxed (S1) | 50 | 2,729,450 | 24.956 | 0.9261 |
| Post-fix, converged (S2) | 101 | 2,729,450 | 24.902 | 0.9267 |
| **Session 3** | **96 (best@76)** | **2,308,723** | **25.364** | **0.9289** |

**+0.250 dB over baseline, +0.462 over session 2, with 15.4% FEWER parameters.**

### The miss, recorded plainly
I predicted F5 (the missing global colour pathway) would be the dominant win. Measured on the
new model, the oracle headroom is **still +3.358 dB** — essentially untouched. The module is
active but converged to a near-constant tone adjustment (gain 1.250/1.255/1.258 with std
0.022/0.025/0.033), despite the same 4,550 parameters capturing 91.7% of the headroom when
trained directly against the oracle in check 4. The pathway is expressive enough; the
end-to-end loss does not drive it there.

Six changes were bundled with no ablation, so the +0.250 dB cannot be attributed to any one of
them — and the one I expected to dominate demonstrably did not. Most likely sources are the
mundane fixes: GroupNorm removing a measured 0.42 dB train/eval mismatch, the capacity
rebalance, and dropping a redundant loss term. Ablation is next-step #2.

---
---

# SESSION 4 — 2026-08-22 — Claiming the Unclaimed Headroom

**Goal:** make the model actually use the `GlobalColorCorrection` pathway session 3 built for
it, and find out which of session 3's six bundled changes mattered.

## PHASE 0 — State check
- [x] Clean tree at `71942d7`; power plan still High performance; no stray trainers
- [x] `checkpoints/best.pt` = epoch 76, 25.2139 dB (AMP) — matches the session-3 record
- [x] Backed up to `checkpoints/_session4_backup/` (md5 `e36a4fd1…` verified)
- [x] Oracle headroom reconfirmed, no drift:
```
model as-is                                   25.3644   0.9289    0.000
+ oracle per-image per-channel OFFSET         28.7223   0.9398   +3.358
+ oracle per-image per-channel AFFINE         31.1767   0.9530   +5.812
fraction of error energy that is pure per-image per-channel DC: 45.2%
```

## PHASE 1 — `L_dc`, and what it took to get there

### S4-D-001 — L1, not MSE, on measured grounds
`L_dc = |mean(pred, per-image per-channel) − mean(target, …)|`, computed exactly as
`tools/_oracle_dc.py` computes its oracle target, so loss and measurement are apples-to-apples.

L1 over MSE because the DC errors are small (~0.035): `d|x|/dx = 1` while `d(x²)/dx = 2x = 0.070`
there, so **MSE would push ~14× more weakly exactly where the push is needed**.

### S4-D-002 — `lambda_dc = 1.0`, calibrated by gradient, not by guesswork
The brief suggested 0.1–0.2. Measuring the gradient *at the module* says that would have
reproduced session 3's failure exactly:

```
||grad|| at global_correction.predictor from L_dc at weight 1.0 : 3.45e-01
||grad|| at the same params from L1 + SSIM + perceptual         : 2.33e-01

 lambda_dc   ratio vs others   % of objective
       0.3             0.44x             6.6%     <- L_dc still LOSES at the module
       0.7             1.03x            14.1%
       1.0             1.48x            19.0%     <- chosen
       2.0             2.96x            32.0%
parity (ratio 1.0) at lambda_dc = 0.68
```

At 0.3 the old terms still dominate the very parameters L_dc is supposed to move. Chose **1.0**:
1.48× dominance at the module, while the spatial terms still hold 81% of the objective.

---

## The finding that reframes this session: most of the +3.36 dB is NOT reachable

Verification checks 3/4 initially failed in a revealing way — optimising L_dc *directly* cut the
DC error by only 18.4%, and the predicted shift **anti-correlated** with what each image needed
(−0.12 / −0.05 / −0.51). Three hypotheses, each tested and killed:

1. **"The context can't predict the offset."** Ordinary least squares said it can:
   R² = 0.692 / 0.866 / 0.897. Hypothesis apparently refuted.
2. **"The clamp inside `GlobalColorCorrection` is eating the gradient."** Measured: only
   **0.93%** of pixels are clamped, and removing the clamp changed the fit from 18.4% to 16.5%.
   Dead.
3. **Then I distrusted my own R².** That fit put **64 context dimensions through 89 samples** —
   it could not not overfit. Redone properly with ridge regression fit on the 801 training
   images and evaluated on the held-out 89 (`tools/_ctx_cv.py`):

```
                                                 R2_R    R2_G    R2_B
physics_context, HELD-OUT                       0.015   0.104   0.346
physics_context, in-sample (my first test)      0.670   0.847   0.886   <- overfit, not evidence
adding input mean / output mean barely helps
```

**The needed offset is mostly not predictable from the input.** The oracle computes it from the
ground-truth reference, and UIEB's references are *human-retouched* — so a large part of that
offset is a retoucher's choice, not a function of the photograph.

`tools/_achievable_ceiling.py` puts a number on it. Fit the best linear offset predictor on the
801 training images, apply it to the held-out 89:

| condition | PSNR | vs base |
|---|---|---|
| model with the correction removed | 22.594 | — |
| + CONSTANT offset (**what session 3 learned**) | 22.730 | +0.136 |
| + offset PREDICTED from the input (best linear, held-out) | 23.275 | **+0.681** |
| + ORACLE offset (uses ground truth) | 25.387 | +2.793 |

**Only 24.4% of the headroom is achievable from the input at all.** The "+3.36 dB" that has been
quoted since session 3 — by me — is an upper bound that requires the answer sheet.

But this is not a dead end: session 3's module captured **+0.136** of an available **+0.681**,
so roughly **+0.55 dB is genuinely on the table** — more than double session 3's entire +0.25 dB
win. That is what tonight's run targets, and it is the honest number to judge it against.

### S4-D-003 — verification thresholds corrected to a reachable standard
Check 4 originally demanded the predicted shift correlate strongly with the needed offset. Once
the held-out R² was measured, that bar was revealed as unreachable by construction. Rewritten to
test what can actually be achieved: the DC error must fall, and the correction must stop being a
constant. Recording this because moving a threshold after seeing a failure is exactly the kind of
change that deserves to be visible rather than quiet.

Final verification at `lambda_dc = 1.0` — all three PASS:
```
ratio (L_dc / others) at the module : 1.48x        -> L_dc dominates
mean |DC error| 0.03509 -> 0.02863  (18.4% reduction)
predicted SHIFT std  [0.0151,0.0100,0.0148] -> [0.0604,0.0281,0.0657]   (4x more per-image)
```

## PHASE 2 — Trained with `L_dc`, and it did not work
- [x] 2.1 Fresh run, watchdog on, early stopping decided the end
- [x] 2.2 Leak-free `validate.py`
- [x] 2.3 Re-ran `tools/_oracle_dc.py` and the per-image variation check — the real test

```
Early stopping triggered: PSNR did not improve for 20 validation rounds.
Training complete. Best PSNR: 24.8377     (best epoch 55, 75 epochs run)
Watchdog: 0 restarts, 0 incidents.
Held-out fp32: PSNR 24.9822  SSIM 0.9240  UIQM 10.1104  UCIQE 0.3167
```

| measure | session 3 | session 4 | change |
|---|---|---|---|
| held-out PSNR | 25.364 | 24.982 | **-0.382** |
| held-out SSIM | 0.9289 | 0.9240 | -0.0049 |
| oracle headroom remaining | +3.358 | +3.192 | -0.166 |
| DC share of remaining error | 45.2% | 44.4% | -0.8 pts |
| predicted gain std | [.022,.025,.033] | [.022,.017,.037] | ~unchanged |

`L_dc` bought 0.166 dB of headroom and cost 0.382 dB of PSNR. The module still converged to a
near-constant correction. Exactly what the Phase-1 measurement predicted: a stronger loss
cannot create information the input does not contain.

### S4-D-004 — restored session 3's checkpoint as canonical
Session 4's model is worse, so leaving it as `checkpoints/best.pt` would quietly degrade the
project. Restored session 3's (verified back at **25.3644 dB / 0.9289**) and preserved
tonight's at `checkpoints/_session4_result/`. Also set `lambda_dc: 0.0` in the config so it
reproduces the installed checkpoint rather than the worse one — a config that silently rebuilds
a rejected model is a trap. The `_dc_loss` implementation stays in `models/loss.py`, working and
documented, for any future experiment.

## PHASE 3 — Ablation: NOT DONE
Each arm is a full training run. The GPU was power-throttled to ~19 W of 77 W all night
(`ChatGPT.exe` co-client again; re-applied the power fix, left the user's app alone), so epochs
took ~146 s instead of ~50 s and one arm is ~3 hours. No room after the main run. It is
next-step #1 and the project should not accept another bundled change before it is done.

## PHASE 4/5 — Report and visual proof
- [x] `FINAL_REPORT_SESSION4.md`
- [x] `outputs/session4_proof.html` — self-contained (1.2 MB, 52 embedded images), opens by
      double-click. 8 rows chosen as 4 best / 2 typical / 2 worst so losses are visible, plus 3
      zoomed crops on the strongest colour casts. Verified it states plainly that tonight's
      change did not work, names session 3 as best-and-installed, and marks session 3 as the
      hero panel rather than letting "newest" imply "best".

### Session verdict
The intervention was correctly designed, correctly verified, and net negative. The finding
worth keeping is the one measured before the run: **the +3.2 dB that has been quoted since
session 3 — by me — is an oracle bound, and only 24.4% of it was ever reachable.** Retiring a
wrong target is worth more than another 0.1 dB.

---
---

# SESSION 6 — 2026-08-22 — LSUI union pretrain -> UIEB fine-tune

**Question:** does more/varied training data help this architecture? Single variable: training
data. Everything else (model, loss, split, evaluation) held fixed.

## PHASE 0 — Fetch, lay out, gate
- [x] 0.1 Clean tree at `06a6e38`; backed up to `checkpoints/_session6_backup/` (md5 `e36a4fd1…`)
- [x] 0.2 **ChatGPT.exe: 10 processes found and force-killed** (S6-D-001)
- [x] 0.3 LSUI laid out from the already-verified archive
- [x] 0.4 **LEAK GATE: PASS**
- [x] 0.5 Held-out split asserted identical to prior sessions
- [x] 0.6 Commit

### S6-D-001 — ChatGPT.exe, standing authorization exercised
Per this session's standing instruction, checked for the GPU co-client that has throttled every
prior run. Found **ten** processes (PIDs 2352, 5468, 20692, 21404, 21708, 22300, 22436, 23580,
27412, 27876) despite no visible window. `CloseMainWindow()` did not clear them, so
`taskkill /IM ChatGPT.exe /F` was used. Afterwards `nvidia-smi --query-compute-apps` returns
**empty** and the power scheme is still High performance. This will be re-checked at every
watchdog check-in, per instruction.

### LSUI layout
Reused `_dsprobe/LSUI.zip` from the feasibility session rather than re-downloading — verified
first: size 492,658,225 bytes (matches the recorded figure) and `testzip()` reports no corrupt
entries. Extracted with an `lsui_` prefix:
```
extracted: input=4279  GT=4279
filename intersection (what dataset.py pairs on): 4279
unpaired : input-only=0  GT-only=0        -> PASS: perfect 1:1 pairing
```
Spot-checked pairs are a genuine enhancement direction (cast 0.505 -> 0.013, 1.003 -> 0.006).
498 MB on disk, inside the gitignored `datasets/`.

### The hard gate — run against the EXTRACTED directory, not the archive
The tool previously scanned `LSUI.zip`. Checking the archive proves the archive is clean;
checking the laid-out directory proves *the files training will actually open* are clean. Added
a directory mode and an explicit exit-code gate, then ran it:
```
scanned 4279 images
closest dHash distance found anywhere : 1 bits
strong hits within 6 bits   : 15  (of which held-out: 0)
closest pair: LSUI/input/lsui_2061.jpg <-> 917_img_.png  1 bit  corr 0.995  [TRAIN split]
LSUI(dir)  held-out strong hits: 0  -> CLEAN
GATE EXIT CODE: 0
```
Same answer as the archive scan. The single 1-bit duplicate is against the *training* split,
which is harmless, and its presence is useful: it shows the detector is live rather than
trivially returning zero.

### Held-out split unchanged — asserted, not assumed
```
train 801 / held-out 89
first 8 held-out: ['708_img_.png','493_img_.png','497_img_.png','784_img_.png',
                   '5554.png','287_img_.png','32_img_.png','573_img_.png']
counts match prior sessions : True
first-8 identical to session 1 : True
train and held-out disjoint : True
SHA256 of held-out list: 0084cf26790978cd5f7ef60ffb958a55ac53892c3dc5773098ff78de5d92a67e
```
The SHA is recorded so future sessions can assert against one value instead of eyeballing.

## PHASE 1 — Two-stage training pools
- [x] 1.1 `data/dataset.py::get_splits()` gained `extra_train_sources` (additive, minimal)
- [x] 1.2 Two stage configs written
- [x] 1.3 Both pools verified programmatically before launching anything

```
stage-1 UNION pool     : 5080 pairs  (801 UIEB + 4279 LSUI)
stage-2 FINE-TUNE pool : 801 pairs
held-out (both stages) : 89 / 89
validation set identical across stages : True
held-out leaked into stage-1 pool      : 0
held-out leaked into stage-2 pool      : 0
duplicate filenames within union pool  : 0
```
Pairs load correctly from *both* halves of the concatenated pool (checked indices 0, 400, 801
and 5079, plus a shuffled DataLoader batch spanning the boundary).

### S6-D-002 — `--init-from` added, because `--resume` would have wrecked the fine-tune
Stage 2 starts from stage 1's weights. The obvious way to do that is `--resume`, and it would
have quietly ruined the run: `--resume` restores `global_step`, and `train.py` computes the LR
as a cosine over `global_step / total_steps`. Stage 1 ends near step 19,050 while stage 2's
schedule is only 4,000 steps long, so the cosine would be driven far past its end and oscillate
back **up**:
```
step 19050 -> cosine progress 7.4 -> lr 7.58e-05
step 20000 -> cosine progress 7.8 -> lr 1.81e-04     (base_lr is 2.00e-04)
```
A "fine-tune" that silently runs at 90% of the from-scratch learning rate would have destroyed
the point of the exercise, and the symptom — a bad stage-2 number — would have looked like
"the mitigation doesn't work" rather than "the LR was wrong".

So `--init-from` loads **model weights only** and starts a genuinely fresh run (epoch 0, step 0,
new optimizer, new schedule). The two flags are mutually exclusive and the code says why.

### S6-D-003 — stage hyper-parameters, and why
**Stage 1 (union):** 30 epochs, not 150. At 5,080 pairs an epoch is 635 steps versus session 3's
100, so **30 union epochs is about 190 UIEB-epochs' worth of gradient steps** — a *longer*
exposure than session 3's 96 epochs, not a shorter one. Warmup 2 epochs (1,270 steps),
patience 10.

**Stage 2 (fine-tune):** lr **2.0e-5**, ten times lower than from-scratch. The job is
re-calibrating an already-converged model toward UIEB's colour convention: a full-rate LR would
wash out the representations learned from 6.3x the data, and a much smaller one would not shift
the calibration at all. 40 epochs of 100 steps, warmup 1, patience 12.

## PHASE 2 — Stage 1: pretrain on the union

### S6-D-004 — ChatGPT.exe was NOT the root cause of the throttling  [CORRECTS THIS SESSION'S PREMISE]
The brief states the throttling has been "traced every time to ChatGPT.exe running as a
background GPU co-client". Killing all 10 of its processes did help, and produced the best
power figure any session has recorded:

```
epoch 1-2, ChatGPT.exe gone : 87.76 W / 2535 MHz   ~340 s/epoch   (previous best ever: 79 W)
```

**But the clamp came back anyway at epoch 3**, with ChatGPT.exe absent and this training run the
only GPU compute client:

```
epoch 3 : 19.33 W / 705 MHz / 55 C / util 99% / reasons 0x4 (SwPowerCap)   629 s/epoch
```

Re-applied the full fix and re-measured — no recovery:
```
powercfg /setactive <High performance>  exit 0
powercfg /overlaysetactive OVERLAY_SCHEME_MAX  exit 0
Win32_Battery BatteryStatus = 2 (on AC)
after 60 s: 19.31 W / 840 MHz / 55 C / SwPowerCap
```

So: High performance scheme active, overlay maxed, on mains, card cool at 55 C, sole GPU
client — and still clamped to a quarter of its power budget. **The cause is the laptop's own
firmware/EC sustained-power policy, which engages after two to three epochs of continuous load
regardless of what software is running.** Killing ChatGPT.exe is still worth doing (it bought
the highest clocks yet, for two epochs) but it is not the fix, and this project should stop
treating it as one. A real fix would need the OEM power utility or a vendor BIOS setting —
outside what an unattended session should touch.

### S6-D-005 — stage 1 cut 30 -> 20 -> 15 epochs, landing on step-parity
Epoch time kept degrading as the clamp tightened — 340 s, then 629 s, then 953 s — so the
horizon was cut twice. It settled at **15 epochs = 9,525 gradient steps**, which is
**step-parity with session 3's 9,600** (96 x 100).

That is not a compromise, it is the better experiment. Identical optimisation budget, 6.3x more
varied data: any difference in the result is attributable to the *data*, not to compute. The
earlier 20-epoch plan (1.32x the steps) would have confounded the two.

No progress was lost either time: the cosine progress was checked against the new horizon
*before* acting (5.6% at the first cut, 15.4% at the second), so the schedule re-anneals
correctly instead of distorting. This is the manoeuvre session 2 used, and the opposite of the
`--resume`-into-a-short-schedule trap S6-D-002 exists to avoid.
```
Resumed from epoch 4 (global_step=2540, best_psnr=22.0741, es_counter=2)
Config: epochs=15   cosine progress 15.4% -> anneals correctly
```

### S6-D-006 — a self-terminating kill script
The script written to stop the trainer matched processes by a regex against their command line.
Its own command line contained the literal string `train.py` (inside the regex), so it matched
**itself** and died before editing anything — leaving the trainer running on the old config.
Same self-matching class of bug the process census caught in session 2. Fixed by killing the
known PID explicitly rather than pattern-matching. Noting it because it silently produced a
half-applied state that looked, from the exit code alone, like the edit had happened.

### Stage 1 result — union pretrain complete
```
Training complete. Best PSNR: 24.4729      (15/15 epochs, 9,525 steps, no early stop)
Best checkpoint: checkpoints/_stage1/best.pt -> copied to checkpoints/_stage1_union.pt
Held-out fp32: PSNR 24.8060  SSIM 0.9177  UIQM 9.8685  UCIQE 0.3126
```
Validation trajectory: 22.40 → 22.37 → 22.07 → 22.19 → 23.25 → 22.49 → 23.70 → 23.94 → 23.73
→ 24.01 → 24.38 → 24.43 → 24.44 → **24.47**. Still climbing at the horizon — it did not early-stop
and had not plateaued, so this figure is a floor, not a converged value.

**24.806 dB, versus session 3's 25.364 on the same held-out 89.** Training on 6.3x more data,
at matched gradient steps, is currently **0.56 dB worse**. That is exactly what the feasibility
report's style-mismatch warning predicted would happen without the mitigation, which is what
stage 2 now tests.

## PHASE 3 — Stage 2: fine-tune back to UIEB
Launched from stage-1 weights with `--init-from` (not `--resume`), UIEB-train alone, lr 2.0e-5:
```
Config: epochs=40  bs=8  lr=2.00e-05
Dataset split: 801 train / 89 validation images
Initialising weights from: checkpoints/_stage1_union.pt (fresh optimizer and LR schedule)
Fine-tune mode: starting at epoch 0, step 0, lr=2.00e-05
```
No `Resumed` line, confirming the fresh schedule S6-D-002 called for.

### Stage 2 result — fine-tune complete
```
Early stopping triggered: PSNR did not improve for 12 validation rounds.
Training complete. Best PSNR: 25.0639     (best epoch 9, 21 epochs run)
Held-out fp32: PSNR 25.3336  SSIM 0.9262  UIQM 10.0170  UCIQE 0.3138
```

## PHASE 4 — Evaluate and decide

| Model | Pairs | Steps | PSNR | SSIM |
|---|---|---|---|---|
| Pre-fix baseline | 801 | ~11,500 | 25.114 | 0.9281 |
| **Session 3 — stays installed** | **801** | **9,600** | **25.364** | **0.9289** |
| S6 stage 1 (union) | 5,080 | 9,525 | 24.806 | 0.9177 |
| S6 stage 2 (union + UIEB FT) | 5,080 -> 801 | 9,525 + 900 | 25.334 | 0.9262 |

stage 2 vs session 3: **-0.031 dB / -0.0027 SSIM**. The fine-tune recovered +0.528 dB of
stage 1's deficit but did not clear the bar. **Decision: session 3's checkpoint stays installed**
(md5-verified untouched — the stage runs wrote to `_stage1/` and `_stage2/` subdirectories), and
`configs/train.yaml` is left alone so it still reproduces the installed model.

### S6-D-007 — the mitigation was aimed at a risk that never fired  [THE FINDING]
The two-stage design existed solely to defuse the feasibility report's style-mismatch warning.
Measured on the held-out 89:

| | R/B | distance from UIEB's target 0.779 |
|---|---|---|
| session 3 (UIEB only) | 0.802 | 0.023 |
| **stage 1 (union, NO fine-tune)** | **0.780** | **0.001** |
| stage 2 (union + UIEB FT) | 0.771 | 0.007 |
| LSUI GT (the other convention) | 0.857 | — |

**Training on the union did not drift the model warm.** It produced *better* colour calibration
than session 3 — 0.001 from target versus 0.023, an order of magnitude closer. And stage 2's
fine-tune moved calibration slightly *away* (0.001 -> 0.007) while gaining 0.53 dB, so whatever
the fine-tune fixed, it was not colour.

So stage 1's 0.56 dB deficit has a different cause. The leading candidate is **exposure, not
step count**: at step-parity each UIEB image was seen ~15 times instead of session 3's ~96.
Step-parity is not exposure-parity, and that is directly testable (report §5).

Recording this as the session's main result. A small win would have been less informative than
ruling out the explanation everyone expected — including the feasibility report, and me.

---
---

# SESSION 7 — 2026-08-22 — Honest positioning, report, website, PDF

No training, no architecture changes. Making six sessions of work legible, honest and
presentable, with real literature grounding.

## PHASE 0 — State check
- [x] Clean tree at `7fcbaff`; best.pt = session 3's model (epoch 76); detector mAP@0.5 = 0.8292
- [x] ChatGPT.exe absent; GPU idle

## PHASE 1 — Novelty assessment -> `docs/novelty_assessment.md`

### S7-D-001 — the honest verdict: nothing here is architecturally novel
Searched 2023-2026 literature and read candidates closely enough to compare *mechanisms*, not
titles. All three of this project's supposed contributions are already published:

**Physics-guided attention.** Sanchez-Ferreira et al. (J. Imaging 12(5):186) encode a physical
prior as "a spatial bias matrix that directly modulates attention affinity" — the same mechanism
class as this project's `Softmax(QKᵀ/√d + λP)V`, for underwater deblurring. PCAFA-Net (Sensors
2025), PGANet (Comput. Electr. Eng. 2025), SFormer (arXiv:2508.18664), physical-guided
transformer interaction (Displays 2023) and a physics-aware diffusion transformer
(arXiv:2403.01497) all occupy the same space. **`docs/math.md` calls this "the core novelty of
the proposed method"; that claim is not supportable and should be rewritten.**

**Wavelet/frequency-split transformer for UIE.** MixRformer (Sensors 25(11):3302) is a
dual-branch wavelet-domain UIE network — structurally the same idea. U-ENHANCE (ACCV 2024 W),
a Mamba spectral-attentive wavelet net (EAAI 2024), WEDM and WWE-UIE all do wavelet+attention
for UIE. Standard practice in this subfield.

**Enhancement hurts detection.** Awad et al., *Beneath the Surface* (arXiv:2411.14626, 2024) ran
**nine** enhancement models x **two** datasets x **three** detectors, **including retraining
detectors on enhanced data**, and found enhancement harms detection at dataset level while
helping individual images. That is the larger version of session 2's experiment, published
first. Session 2's result is a faithful **reproduction**, not a discovery, and will be presented
as such.

### What survives as contribution
1. **The correctness fix and its falsification methodology** — the strongest item. The original
   attention had no Q/K/V projections, making it provably incapable of a global colour shift
   (`verify_attention.py`: MSE 9.000 = 3.0², output mean unmoved). Diagnosing that and proving
   it by falsification is real engineering rigour; the fixed module itself is textbook attention.
2. **Controlled negative results** — attention fix didn't move PSNR; `L_dc` made it worse; 6.3x
   data didn't help; the predicted colour-style mismatch never occurred.
3. **Flagged but NOT claimed:** session 4's oracle-vs-achievable decomposition (only 24.4% of
   the apparent colour headroom is reachable, because UIEB references are human-retouched and
   the offset is weakly predictable, held-out R² 0.015/0.104/0.346). Not found in the searches
   run, but "I didn't find it" is weak evidence and it is recorded as needing a proper check.

### S7-D-002 — our 25.364 dB is NOT comparable to published UIEB numbers
Published Test-U90 figures: WaterNet 19.81, FUnIE 19.45, UGAN 20.68, Ucolor 20.78, U-shape
Transformer 22.91/0.91; PCAFA-Net 22.80/0.890 on UIEB. Our 25.364 is on **our own seed-42 split
of 89 images**, not the community's standard Test-U90/C90. Different image sets, and UIEB
difficulty varies enormously. A 2.3 M-parameter model beating a published transformer by 2.4 dB
almost certainly reflects an easier split, not a better method. The report presents this as
context and explicitly refuses the "we beat X" reading.
