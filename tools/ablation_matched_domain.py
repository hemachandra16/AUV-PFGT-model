"""ABLATION part 2: matched-domain test — does enhancement help once the detector adapts?

Part 1 (`ablation_enhance_detect.py`) showed the deployed pipeline loses 3.9 mAP50 points
when a RAW-trained detector is fed enhanced frames. That result is confounded: it cannot
separate "enhancement destroys useful information" from "the detector was never trained on
this distribution".

A detail probe over 300 paired frames ruled out the first explanation — enhanced frames are
*sharper*, not blurrier (Laplacian variance +18%, high-pass energy +39%, contrast +40%). So
the loss is almost certainly domain shift, and the decisive test is to train the detector on
enhanced frames and evaluate it on enhanced frames.

This runs BOTH arms at an identical, reduced training budget so the comparison is fair:

    arm RAW      : fine-tune YOLO11n on N raw frames      -> evaluate on raw val
    arm ENHANCED : fine-tune YOLO11n on N enhanced frames -> evaluate on enhanced val

Same N, same epochs, same seed, same hyperparameters. The only difference is the image
domain. A reduced N is used deliberately: the point is the *difference between arms*, not
the absolute mAP, and matching budgets matters more than maximising either.

Usage:
    python tools/ablation_matched_domain.py --n-train 3000 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "datasets" / "RUOD_yolo"
ENH_VAL = ROOT / "datasets" / "RUOD_yolo_enhanced"
WORK = ROOT / "datasets" / "_ablation_matched"

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


def write_yaml(path: Path, root: Path, train_rel: str, val_rel: str) -> Path:
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    path.write_text(
        f"path: {root.as_posix()}\n"
        f"train: {train_rel}\n"
        f"val: {val_rel}\n\n"
        "names:\n" + names_block + "\n",
        encoding="utf-8",
    )
    return path


def enhance_images(enhancer, names, src_img: Path, dst_img: Path,
                   device: str, max_side: int) -> float:
    dst_img.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    done = 0
    for name in names:
        out_path = dst_img / name
        if out_path.exists():
            done += 1
            continue
        with Image.open(src_img / name) as im:
            im = im.convert("RGB")
            w0, h0 = im.size
            if max_side and max(w0, h0) > max_side:
                s = max_side / max(w0, h0)
                proc = im.resize((int(w0 * s), int(h0 * s)), Image.Resampling.BILINEAR)
            else:
                proc = im
            arr = np.array(proc, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        _, _, h, w = t.shape
        ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
        if ph or pw:
            t = F.pad(t, (0, pw, 0, ph), mode="reflect")
        with torch.no_grad():
            y = enhancer(t)
        y = y[:, :, :h, :w]
        out = (y.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        img = Image.fromarray(out)
        if img.size != (w0, h0):
            img = img.resize((w0, h0), Image.Resampling.BILINEAR)
        img.save(out_path, quality=95)
        done += 1
        if done % 300 == 0:
            el = time.time() - t0
            print(f"    {done}/{len(names)}  {el/done:.2f}s/img  "
                  f"ETA {(len(names)-done)*el/done/60:.1f} min", flush=True)
    return time.time() - t0


def train_and_eval(tag: str, data_yaml: Path, epochs: int, imgsz: int, batch: int) -> dict:
    from ultralytics import YOLO
    print(f"\n=== ARM {tag}: fine-tuning YOLO11n ===", flush=True)
    model = YOLO("yolo11n.pt")
    res = model.train(
        data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
        workers=4, patience=epochs, project=str(ROOT / "runs" / "ablation"),
        name=f"matched_{tag}", device=0, exist_ok=True, plots=False, val=True,
        seed=42, verbose=False,
    )
    best = Path(res.save_dir) / "weights" / "best.pt"
    m = YOLO(str(best)).val(data=str(data_yaml), imgsz=imgsz, device=0,
                            verbose=False, plots=False)
    box = m.box
    names = m.names if isinstance(m.names, dict) else {}
    per_class = {}
    try:
        for i, ci in enumerate(box.ap_class_index):
            per_class[names.get(int(ci), str(ci))] = round(float(box.ap50[i]), 4)
    except Exception:
        pass
    out = {
        "arm": tag, "weights": str(best),
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "per_class_AP50": per_class,
    }
    print(f"  [{tag}] mAP50={out['mAP50']:.4f}  mAP50-95={out['mAP50_95']:.4f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--enhancer", default="checkpoints/best.pt")
    ap.add_argument("--out", default="results/ablation_matched_domain.json")
    args = ap.parse_args()

    device = "cuda"
    train_names = sorted(p.name for p in (SRC / "images" / "train").glob("*.jpg"))[: args.n_train]
    val_names = sorted(p.name for p in (SRC / "images" / "val").glob("*.jpg"))
    print(f"matched-domain ablation: {len(train_names)} train / {len(val_names)} val, "
          f"{args.epochs} epochs per arm\n")

    # ---------- RAW arm dataset ----------
    raw_root = WORK / "raw"
    for split, names in (("train", train_names), ("val", val_names)):
        (raw_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (raw_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for n in names:
            link_or_copy(SRC / "images" / split / n, raw_root / "images" / split / n)
            link_or_copy(SRC / "labels" / split / f"{Path(n).stem}.txt",
                         raw_root / "labels" / split / f"{Path(n).stem}.txt")
    raw_yaml = write_yaml(raw_root / "raw.yaml", raw_root, "images/train", "images/val")

    # ---------- ENHANCED arm dataset ----------
    enh_root = WORK / "enhanced"
    (enh_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (enh_root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    (enh_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    for n in train_names:
        link_or_copy(SRC / "labels" / "train" / f"{Path(n).stem}.txt",
                     enh_root / "labels" / "train" / f"{Path(n).stem}.txt")
    # Val frames were already enhanced by part 1 — reuse them rather than redo the work.
    for n in val_names:
        link_or_copy(ENH_VAL / "images" / "val" / n, enh_root / "images" / "val" / n)
        link_or_copy(SRC / "labels" / "val" / f"{Path(n).stem}.txt",
                     enh_root / "labels" / "val" / f"{Path(n).stem}.txt")

    print("Enhancing training frames (val frames reused from part 1) ...")
    from models.build import build_model
    from utils.checkpoint import load_checkpoint
    enhancer = build_model(device=device)
    load_checkpoint(args.enhancer, model=enhancer, device=torch.device(device))
    enhancer.eval()
    secs = enhance_images(enhancer, train_names, SRC / "images" / "train",
                          enh_root / "images" / "train", device, args.max_side)
    print(f"  enhanced {len(train_names)} train frames in {secs/60:.1f} min")
    del enhancer
    torch.cuda.empty_cache()

    enh_yaml = write_yaml(enh_root / "enhanced.yaml", enh_root, "images/train", "images/val")

    raw_res = train_and_eval("raw", raw_yaml, args.epochs, args.imgsz, args.batch)
    enh_res = train_and_eval("enhanced", enh_yaml, args.epochs, args.imgsz, args.batch)

    d50 = enh_res["mAP50"] - raw_res["mAP50"]
    d5095 = enh_res["mAP50_95"] - raw_res["mAP50_95"]
    per_delta = {c: round(enh_res["per_class_AP50"].get(c, 0) - raw_res["per_class_AP50"].get(c, 0), 4)
                 for c in CLASS_NAMES
                 if c in raw_res["per_class_AP50"] and c in enh_res["per_class_AP50"]}

    summary = {
        "question": "With the detector trained on the SAME domain it is tested on, does "
                    "PFGT-UIE enhancement help underwater detection?",
        "design": "Identical budget both arms: same N, epochs, imgsz, batch, seed. Only the "
                  "image domain differs.",
        "n_train": len(train_names), "n_val": len(val_names),
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "raw_arm": raw_res, "enhanced_arm": enh_res,
        "delta_mAP50": round(d50, 4), "delta_mAP50_95": round(d5095, 4),
        "per_class_delta_AP50": per_delta,
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 68)
    print("MATCHED-DOMAIN RESULT (train and test in the same domain, equal budget)")
    print("=" * 68)
    print(f"  raw arm      : mAP50 {raw_res['mAP50']:.4f}   mAP50-95 {raw_res['mAP50_95']:.4f}")
    print(f"  enhanced arm : mAP50 {enh_res['mAP50']:.4f}   mAP50-95 {enh_res['mAP50_95']:.4f}")
    print(f"  delta        : mAP50 {d50:+.4f}   mAP50-95 {d5095:+.4f}")
    print("=" * 68)
    for c, d in sorted(per_delta.items(), key=lambda kv: kv[1], reverse=True):
        print(f"    {c:<14} {d:+.4f}")
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
