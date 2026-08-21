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
