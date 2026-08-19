"""Fine-tune a YOLO detector on RUOD for underwater object detection.

Replaces the previous approach (COCO-pretrained Faster R-CNN with hand-mapped class
names) with a detector actually trained on underwater imagery.

Defaults are sized for an 8 GB RTX 4060 Laptop GPU. On CUDA OOM the script halves the
batch size and retries rather than dying.

Usage:
    python tools/train_detector.py                      # defaults
    python tools/train_detector.py --epochs 30 --batch 24
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "RUOD_yolo" / "ruod.yaml"))
    ap.add_argument("--model", default="yolo11n.pt", help="Base weights to fine-tune")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--time-budget-hours", type=float, default=None,
                    help="Hard wall-clock cap; Ultralytics stops cleanly when reached")
    ap.add_argument("--project", default=str(ROOT / "runs" / "detect"))
    ap.add_argument("--name", default="ruod_yolo11n")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    if not Path(args.data).exists():
        sys.exit(f"Dataset config not found: {args.data}\n"
                 "Run tools/fetch_ruod.py then tools/ruod_to_yolo.py first.")

    print(f"Fine-tuning {args.model} on RUOD")
    print(f"  data      : {args.data}")
    print(f"  epochs    : {args.epochs}   imgsz: {args.imgsz}   batch: {args.batch}")
    if torch.cuda.is_available():
        print(f"  gpu       : {torch.cuda.get_device_name(0)}")

    batch = args.batch
    results = None
    while True:
        try:
            model = YOLO(args.model)
            kwargs = dict(
                data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=batch,
                workers=args.workers,
                patience=args.patience,
                project=args.project,
                name=args.name,
                device=args.device,
                exist_ok=True,
                plots=True,
                val=True,
                seed=42,
            )
            if args.time_budget_hours:
                kwargs["time"] = args.time_budget_hours
            results = model.train(**kwargs)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if batch <= 2:
                sys.exit("CUDA OOM even at batch=2; aborting.")
            batch //= 2
            print(f"\n!! CUDA OOM -> halving batch size to {batch} and retrying\n")

    # ---- Report metrics ----
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"\nBest weights: {best}")

    print("\nRunning final validation for metrics ...")
    metrics = YOLO(str(best)).val(data=args.data, imgsz=args.imgsz, device=args.device,
                                  verbose=False)

    box = metrics.box
    names = metrics.names if isinstance(metrics.names, dict) else {}
    per_class = {}
    try:
        for i, cls_idx in enumerate(box.ap_class_index):
            per_class[names.get(int(cls_idx), str(cls_idx))] = {
                "AP50": round(float(box.ap50[i]), 4),
                "AP50_95": round(float(box.ap[i]), 4),
                "precision": round(float(box.p[i]), 4),
                "recall": round(float(box.r[i]), 4),
            }
    except Exception as exc:  # pragma: no cover
        print(f"(per-class breakdown unavailable: {exc})")

    summary = {
        "dataset": "RUOD (Real-world Underwater Object Detection)",
        "base_model": args.model,
        "epochs_requested": args.epochs,
        "imgsz": args.imgsz,
        "batch": batch,
        "weights": str(best),
        "metrics": {
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
        },
        "per_class": per_class,
    }

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detection_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Publish the weights where models/object_detection.py looks for them by default.
    dest = ROOT / "checkpoints" / "detector"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest / "best.pt")

    print("\n" + "=" * 60)
    print(f"  mAP@0.5      : {summary['metrics']['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95 : {summary['metrics']['mAP50_95']:.4f}")
    print(f"  precision    : {summary['metrics']['precision']:.4f}")
    print(f"  recall       : {summary['metrics']['recall']:.4f}")
    print("=" * 60)
    for name, m in sorted(per_class.items(), key=lambda kv: -kv[1]["AP50"]):
        print(f"  {name:<14} AP50={m['AP50']:.4f}  AP50-95={m['AP50_95']:.4f}")
    print("=" * 60)
    print(f"metrics -> results/detection_metrics.json")
    print(f"weights -> {dest / 'best.pt'}")


if __name__ == "__main__":
    main()
