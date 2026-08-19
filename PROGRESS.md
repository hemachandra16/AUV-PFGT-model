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

## PHASE 1 — Fix core attention bug
- [ ] 1.1 Real Q/K/V projections + multi-head in physics_attention.py
- [ ] 1.2 Upgrade physics bias from rank-1 scalar to projected per-position bias
- [ ] 1.3 models/build.py::build_model() single source of truth; update 8 entry points
- [ ] 1.4 L_frequency loss + lambda_frequency config
- [ ] 1.5 Smoke-train verification (loss decreases, visible output change)
- [ ] 1.6 Commit

## PHASE 2 — Eval pipeline correctness
- [ ] 2.1 Shared seeded 90/10 split in data/dataset.py
- [ ] 2.2 Wire training.save_every_epochs
- [ ] 2.3 Re-verify detection "grid of boxes" does not reproduce
- [ ] 2.4 Commit

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
