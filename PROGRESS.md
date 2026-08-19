# PFGT-UIE — Autonomous Fix + Detection + Overnight Training Run

**Session start:** 2026-08-20 ~00:15 local
**Operator:** unattended (bypass-permissions). All decisions made autonomously and logged here.
**Machine:** Ryzen 7 7840HS / 16 GB RAM / RTX 4060 Laptop 8 GB (sm_89, Ada) / driver 592.82 (CUDA up to 13.1) / 135 GB free on D:

Legend: `[ ]` pending · `[~]` in progress · `[x]` done · `[!]` blocked / worked around

---

## PHASE 0 — Safety net & environment
- [~] 0.1 git init + .gitignore + baseline commit
- [ ] 0.2 Verify GPU (nvidia-smi)
- [ ] 0.3 Test copied .venv
- [ ] 0.4 Install stable PyTorch (NOT nightly) matched to CUDA
- [ ] 0.5 Install remaining deps
- [ ] 0.6 Rewrite requirements.txt as UTF-8 from working venv
- [ ] 0.7 Byte-diff + archive stray files (nested dup folder, models.zip, losses/)
- [ ] 0.8 Back up checkpoints to checkpoints/_baseline_before_fixes/
- [ ] 0.9 Run smoke tests on unmodified model
- [ ] 0.10 Commit

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
