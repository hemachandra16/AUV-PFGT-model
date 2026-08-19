"""End-to-end detection evidence: PFGT-UIE enhancement -> underwater detector.

Produces, for each sample image, a side-by-side panel of

    raw | enhanced | enhanced + fine-tuned RUOD YOLO | enhanced + legacy COCO Faster R-CNN

and prints the detection counts and class names each detector reports. This covers three
things at once:

  * Phase 3 evidence — the fine-tuned detector emits real marine class names on real
    underwater frames, rather than COCO labels renamed by a lookup table.
  * Phase 2.3 re-check — confirms the historical "grid of hundreds of overlapping boxes at
    ~49% confidence" symptom does not reproduce. A sane detector returns a handful of
    boxes; the check fails loudly if either backend returns an absurd number.
  * A like-for-like comparison of the two backends on identical inputs.

Usage:
    python tools/detection_check.py --checkpoint checkpoints/best.pt --n 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# More than this many boxes on a single frame means something is wrong (e.g. NMS not
# running). The historical bug produced hundreds.
SANE_BOX_LIMIT = 60


def load_rgb(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def enhance(model, tensor: torch.Tensor, device: str) -> torch.Tensor:
    _, _, h, w = tensor.shape
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    x = tensor.to(device)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        y = model(x)
    return y[:, :, :h, :w].cpu()


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def banner(img: Image.Image, text: str, width: int) -> Image.Image:
    img = img.resize((width, int(img.height * width / img.width)))
    out = Image.new("RGB", (img.width, img.height + 24), (18, 18, 22))
    out.paste(img, (0, 24))
    ImageDraw.Draw(out).text((6, 7), text, fill=(235, 235, 240))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out-dir", default="outputs/_phase3_check")
    ap.add_argument("--skip-legacy", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from data.dataset import get_splits, subset_pair_names
    from models.build import build_model
    from models.object_detection import build_detector, annotate_image_with_detections
    from utils.checkpoint import load_checkpoint

    device = args.device
    raw_dir = ROOT / "datasets" / "UIEB" / "raw-890"
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sample from the held-out split so nothing here was trained on.
    _, val_subset = get_splits(augment_train=False)
    names = subset_pair_names(val_subset)[: args.n]

    print("Loading PFGT-UIE enhancer ...")
    enhancer = build_model(device=device)
    load_checkpoint(args.checkpoint, model=enhancer, device=torch.device(device))
    enhancer.eval()

    print("Loading fine-tuned RUOD detector ...")
    yolo = build_detector(backend="yolo", conf_threshold=args.conf, iou_threshold=args.iou)

    legacy = None
    if not args.skip_legacy:
        print("Loading legacy COCO Faster R-CNN for comparison ...")
        try:
            legacy = build_detector(backend="fasterrcnn", conf_threshold=0.45,
                                    iou_threshold=args.iou)
            legacy.to(device).eval()
        except Exception as exc:
            print(f"  (legacy detector unavailable: {exc})")

    report = {"conf_threshold": args.conf, "iou_threshold": args.iou, "samples": []}
    sane = True

    for name in names:
        raw_t = load_rgb(raw_dir / name)
        enh_t = enhance(enhancer, raw_t, device)
        raw_img, enh_img = to_pil(raw_t), to_pil(enh_t)

        y_dets = yolo.detect_objects(enh_t, conf_threshold=args.conf, iou_threshold=args.iou)
        y_classes = sorted({d["class"] for d in y_dets})

        l_dets, l_classes = [], []
        if legacy is not None:
            l_dets = legacy.detect_objects(enh_t, conf_threshold=0.45, iou_threshold=args.iou)
            l_classes = sorted({d["class"] for d in l_dets})

        print(f"\n{name}")
        print(f"  RUOD YOLO (fine-tuned) : {len(y_dets):3d} boxes  classes={y_classes}")
        if legacy is not None:
            print(f"  COCO Faster R-CNN (old): {len(l_dets):3d} boxes  classes={l_classes}")
            print(f"                           raw COCO labels={sorted({d['raw_class'] for d in l_dets})}")

        for label_, dets in (("yolo", y_dets), ("legacy", l_dets)):
            if len(dets) > SANE_BOX_LIMIT:
                print(f"  !! {label_} returned {len(dets)} boxes (> {SANE_BOX_LIMIT}) "
                      f"-- possible NMS failure")
                sane = False

        panels = [banner(raw_img, "raw input", 420),
                  banner(enh_img, "PFGT-UIE enhanced", 420),
                  banner(annotate_image_with_detections(enh_img.copy(), y_dets),
                         f"RUOD YOLO (fine-tuned): {len(y_dets)} boxes", 420)]
        if legacy is not None:
            panels.append(banner(annotate_image_with_detections(enh_img.copy(), l_dets),
                                 f"LEGACY COCO F-RCNN: {len(l_dets)} boxes", 420))

        h = max(p.height for p in panels)
        grid = Image.new("RGB", (sum(p.width for p in panels), h), (18, 18, 22))
        x = 0
        for p in panels:
            grid.paste(p, (x, 0)); x += p.width
        out_path = out_dir / f"detect_{Path(name).stem}.png"
        grid.save(out_path)

        report["samples"].append({
            "image": name,
            "yolo_boxes": len(y_dets),
            "yolo_classes": y_classes,
            "yolo_detections": [{"class": d["class"], "confidence": round(d["confidence"], 3),
                                 "bbox": d["bbox"]} for d in y_dets],
            "legacy_boxes": len(l_dets),
            "legacy_classes": l_classes,
            "panel": str(out_path.relative_to(ROOT)),
        })

    report["max_boxes_any_sample"] = max(
        [s["yolo_boxes"] for s in report["samples"]] +
        [s["legacy_boxes"] for s in report["samples"]] + [0]
    )
    report["grid_of_boxes_bug_reproduced"] = not sane

    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "detection_samples.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"max boxes on any sample: {report['max_boxes_any_sample']} "
          f"(sane limit {SANE_BOX_LIMIT})")
    print(f"'grid of hundreds of boxes' bug reproduced: {report['grid_of_boxes_bug_reproduced']}")
    print(f"panels -> {out_dir}")
    print(f"report -> results/detection_samples.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
