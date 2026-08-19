"""Convert the RUOD dataset from COCO format to the YOLO layout Ultralytics expects.

RUOD (Real-world Underwater Object Detection) — 14,000 real underwater images over 10
marine classes: holothurian, echinus, scallop, starfish, fish, corals, diver,
cuttlefish, turtle, jellyfish.

Source layout (as published on the HF Hub at Mortallll/RUOD):
    coco/annotations/instances_train.json
    coco/annotations/instances_val.json
    coco/train/*.jpg
    coco/val/*.jpg

Target layout:
    datasets/RUOD_yolo/images/{train,val}/*.jpg      (symlinked, or copied on failure)
    datasets/RUOD_yolo/labels/{train,val}/*.txt      (class cx cy w h, normalised)
    datasets/RUOD_yolo/ruod.yaml

Images are symlinked rather than copied so the conversion costs no extra disk and takes
seconds instead of minutes. Ultralytics follows symlinks fine. If symlink creation fails
(Windows without Developer Mode / admin), it silently falls back to copying.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CLASS_NAMES = [
    "holothurian", "echinus", "scallop", "starfish", "fish",
    "corals", "diver", "cuttlefish", "turtle", "jellyfish",
]


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError, AttributeError):
        shutil.copy2(src, dst)


def convert_split(
    ann_path: Path,
    image_root: Path,
    out_images: Path,
    out_labels: Path,
    subset: int | None,
    seed: int,
) -> dict:
    with open(ann_path, encoding="utf-8") as handle:
        coco = json.load(handle)

    # COCO category ids are 1..10 and already in CLASS_NAMES order; map to 0-based.
    cat_id_to_idx = {}
    for cat in coco["categories"]:
        name = cat["name"]
        if name not in CLASS_NAMES:
            raise ValueError(f"Unexpected RUOD category {name!r}; expected one of {CLASS_NAMES}")
        cat_id_to_idx[cat["id"]] = CLASS_NAMES.index(name)

    images = {img["id"]: img for img in coco["images"]}

    anns_by_image: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[ann["image_id"]].append(ann)

    image_ids = sorted(images)
    if subset is not None and subset < len(image_ids):
        # Deterministic subsample, so a re-run reproduces the same subset.
        random.Random(seed).shuffle(image_ids)
        image_ids = sorted(image_ids[:subset])

    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    class_counts: Counter = Counter()
    n_images = 0
    n_boxes = 0
    n_missing = 0
    n_degenerate = 0

    for img_id in image_ids:
        info = images[img_id]
        file_name = Path(info["file_name"]).name
        src = image_root / file_name
        if not src.exists():
            n_missing += 1
            continue

        width = float(info["width"])
        height = float(info["height"])

        lines = []
        for ann in anns_by_image.get(img_id, []):
            x, y, w, h = ann["bbox"]
            # Clip to the image, then drop anything with no area left.
            x0 = max(0.0, min(float(x), width))
            y0 = max(0.0, min(float(y), height))
            x1 = max(0.0, min(float(x) + float(w), width))
            y1 = max(0.0, min(float(y) + float(h), height))
            bw, bh = x1 - x0, y1 - y0
            if bw <= 1.0 or bh <= 1.0:
                n_degenerate += 1
                continue
            cx = (x0 + bw / 2.0) / width
            cy = (y0 + bh / 2.0) / height
            idx = cat_id_to_idx[ann["category_id"]]
            lines.append(f"{idx} {cx:.6f} {cy:.6f} {bw / width:.6f} {bh / height:.6f}")
            class_counts[CLASS_NAMES[idx]] += 1
            n_boxes += 1

        link_or_copy(src, out_images / file_name)
        (out_labels / f"{Path(file_name).stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        n_images += 1

    return {
        "images": n_images,
        "boxes": n_boxes,
        "missing_images": n_missing,
        "degenerate_boxes": n_degenerate,
        "class_counts": dict(class_counts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="datasets/RUOD_raw/coco")
    ap.add_argument("--dst", default="datasets/RUOD_yolo")
    ap.add_argument("--train-subset", type=int, default=None,
                    help="Cap the number of training images (None = use all)")
    ap.add_argument("--val-subset", type=int, default=None,
                    help="Cap the number of validation images (None = use all)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = (ROOT / args.src) if not Path(args.src).is_absolute() else Path(args.src)
    dst = (ROOT / args.dst) if not Path(args.dst).is_absolute() else Path(args.dst)

    stats = {}
    for split, ann_name, img_subdir, subset in [
        ("train", "instances_train.json", "train", args.train_subset),
        ("val", "instances_val.json", "val", args.val_subset),
    ]:
        ann_path = src / "annotations" / ann_name
        image_root = src / img_subdir
        if not ann_path.exists():
            sys.exit(f"Missing annotations: {ann_path}")
        if not image_root.exists():
            sys.exit(f"Missing images: {image_root}")

        print(f"Converting {split} ...")
        stats[split] = convert_split(
            ann_path, image_root,
            dst / "images" / split, dst / "labels" / split,
            subset, args.seed,
        )
        s = stats[split]
        print(f"  {split}: {s['images']} images, {s['boxes']} boxes"
              f" (missing {s['missing_images']}, degenerate {s['degenerate_boxes']})")

    yaml_path = dst / "ruod.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        "# RUOD — Real-world Underwater Object Detection (10 marine classes)\n"
        "# Generated by tools/ruod_to_yolo.py\n"
        f"path: {dst.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n" + names_block + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {yaml_path}")

    print("\nClass distribution (train):")
    counts = stats["train"]["class_counts"]
    total = sum(counts.values()) or 1
    for name in CLASS_NAMES:
        c = counts.get(name, 0)
        print(f"  {name:<14} {c:>7,}  ({c / total * 100:5.2f}%)")

    (dst / "conversion_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
