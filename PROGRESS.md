# PFGT-UIE — Autonomous Fix + Detection + Overnight Training Run

**Session start:** 2026-08-20 ~00:15 local
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

## PHASE 3 — Real underwater object detection
- [ ] 3.1 Choose + fetch dataset (RUOD primary, DUO/UTDAC2020 fallback)
- [ ] 3.2 Convert to YOLO format
- [ ] 3.3 Fine-tune YOLO11n/YOLOv8n
- [ ] 3.4 Wire into models/object_detection.py + infer_detection.py
- [ ] 3.5 Report mAP@0.5 to results/detection_metrics.json
- [ ] 3.6 Sample end-to-end outputs to outputs/_phase3_check/
- [ ] 3.7 Commit

## PHASE 4 — Overnight enhancement training run
- [ ] 4.1 Launch background training -> logs/train.log
- [ ] 4.2 Periodic check-ins (every 30-45 min)
- [ ] 4.3 Final leakage-free validate.py
- [ ] 4.4 Sample infer.py outputs -> outputs/_final_check/
- [ ] 4.5 Commit

## PHASE 5 — Final report
- [ ] 5.1 FINAL_REPORT.md
- [ ] 5.2 README/docs cleanup

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
