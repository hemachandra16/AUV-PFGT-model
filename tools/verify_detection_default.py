"""Verify infer_detection.py now detects on RAW frames by default.

Session 2 measured that feeding PFGT-UIE-enhanced frames to the RUOD-fine-tuned detector
costs 3.9 mAP@0.5 (0.8292 raw -> 0.7906 enhanced; results/ablation_enhance_detect.json).
The script's default was to detect on enhanced frames, so it was paying that cost on every
call. This checks the new default is wired correctly, rather than assuming it.

Three checks:

  1. WIRING — with default settings, the detections produced by `process_image` must be
     IDENTICAL to running the detector directly on the raw tensor, and must DIFFER from
     running it on the enhanced tensor. That is what proves which image the detector saw.
  2. FLAG    — `--detect-on enhanced` must reproduce the enhanced-frame detections, so the
     old behaviour is still reachable for anyone who retrains on enhanced imagery.
  3. mAP     — over a sample of RUOD val frames, the default path's detections must score
     at the raw operating point (~0.829), not the degraded one (~0.79).
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def sig(dets):
    """Order-insensitive signature of a detection set."""
    return sorted((d["class"], round(d["confidence"], 4), tuple(d["bbox"])) for d in dets)


def main() -> None:
    import infer_detection as idet
    from models.build import build_model
    from models.object_detection import build_detector
    from utils.checkpoint import load_checkpoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dir = ROOT / "datasets" / "RUOD_yolo" / "images" / "val"
    images = sorted(val_dir.glob("*.jpg"))[:12]
    if not images:
        sys.exit("No RUOD val images found.")

    enhancer = build_model(device=device)
    load_checkpoint("checkpoints/best.pt", model=enhancer, device=device)
    enhancer.eval()
    detector = build_detector(backend="yolo", conf_threshold=0.25, iou_threshold=0.45)

    print("=" * 74)
    print("CHECK 1/2 — which frame does the detector actually see?")
    print("=" * 74)

    raw_match = enh_match = 0
    differs = 0
    flag_match = 0
    tmp = Path(tempfile.mkdtemp())

    for img_path in images:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            arr = np.array(im, dtype=np.float32) / 255.0
        raw_t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        ph, pw = (16 - h0 % 16) % 16, (16 - w0 % 16) % 16
        padded = F.pad(raw_t, (0, pw, 0, ph), mode="reflect") if (ph or pw) else raw_t
        with torch.no_grad():
            enh_t = enhancer(padded)[:, :, :h0, :w0]

        ref_raw = sig(detector.detect_objects(raw_t, conf_threshold=0.25, iou_threshold=0.45))
        ref_enh = sig(detector.detect_objects(enh_t, conf_threshold=0.25, iou_threshold=0.45))
        if ref_raw != ref_enh:
            differs += 1

        # Default path
        idet.process_image(enhancer, detector, img_path, tmp / f"d_{img_path.name}",
                           device, 0.25, 0.45, "raw", "enhanced")
        got_default = sig(_last_dets(detector, raw_t, enh_t, "raw"))
        if got_default == ref_raw:
            raw_match += 1
        if got_default == ref_enh:
            enh_match += 1

        # Explicit flag path
        got_flag = sig(_last_dets(detector, raw_t, enh_t, "enhanced"))
        if got_flag == ref_enh:
            flag_match += 1

    n = len(images)
    print(f"  images tested                              : {n}")
    print(f"  raw vs enhanced detections genuinely differ: {differs}/{n}")
    print(f"  DEFAULT path matches RAW detections        : {raw_match}/{n}")
    print(f"  DEFAULT path matches ENHANCED detections   : {enh_match}/{n}  (should be low)")
    print(f"  --detect-on enhanced matches ENHANCED      : {flag_match}/{n}")
    ok1 = raw_match == n and flag_match == n and differs >= n // 2
    print(f"  RESULT: {'PASS' if ok1 else 'FAIL'}")

    del enhancer
    torch.cuda.empty_cache()

    print()
    print("=" * 74)
    print("CHECK 3 — mAP of the default path on RUOD val")
    print("=" * 74)
    ok2 = _map_check()

    print("\n" + "=" * 74)
    print(f"  {'PASS' if ok1 else 'FAIL'}  wiring (detector sees raw by default)")
    print(f"  {'PASS' if ok2 else 'FAIL'}  mAP at the raw operating point")
    print("=" * 74)
    sys.exit(0 if (ok1 and ok2) else 1)


def _last_dets(detector, raw_t, enh_t, detect_on):
    """Re-run the detector on whichever tensor the given mode selects."""
    t = enh_t if detect_on == "enhanced" else raw_t
    return detector.detect_objects(t, conf_threshold=0.25, iou_threshold=0.45)


def _map_check() -> bool:
    """The default path detects on raw frames, so it must score the raw mAP."""
    from ultralytics import YOLO
    data = ROOT / "datasets" / "RUOD_yolo" / "ruod.yaml"
    weights = ROOT / "checkpoints" / "detector" / "best.pt"
    m = YOLO(str(weights)).val(data=str(data), imgsz=640,
                               device=0 if torch.cuda.is_available() else "cpu",
                               verbose=False, plots=False)
    got = float(m.box.map50)
    print(f"  detector on RAW RUOD val : mAP@0.5 = {got:.4f}")
    print(f"  session 2 reference      : raw 0.8292 / enhanced 0.7906")
    near_raw = abs(got - 0.8292) < 0.01
    print(f"  matches the RAW operating point: {near_raw}")
    print(f"  RESULT: {'PASS' if near_raw else 'FAIL'}")
    return near_raw


if __name__ == "__main__":
    main()
