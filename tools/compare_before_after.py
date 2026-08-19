"""Before/after comparison: old (pre-fix) model vs new (post-fix) model.

The headline symptom of the attention bug was that enhanced output looked like a
near-copy of the hazy input. That is measurable: if the model is close to an identity
map, PSNR(input, output) is very high. A model that genuinely corrects colour and haze
must move *away* from its input and *toward* the reference.

So for each sample this reports two numbers per model:

    PSNR(raw, enhanced)        high  => near-copy of the degraded input (BAD)
    PSNR(reference, enhanced)  high  => close to the ground-truth target (GOOD)

The old checkpoint must be run against the old code, because the fixed attention module
has parameters (q/k/v/out projections, per-head physics projection) that the old
checkpoint does not contain. The old code lives in the git worktree at
``_archive/baseline_code`` (commit ad0f4d8), and is loaded here in a subprocess-free way
by temporarily putting that worktree at the front of sys.path.

Usage:
    python tools/compare_before_after.py --new-checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINE_CODE = ROOT / "_archive" / "baseline_code"


def load_rgb(path: Path, size: int = 256) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return 10.0 * float(np.log10(1.0 / mse))


def to_pil(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze(0).detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _purge_project_modules() -> None:
    """Drop cached project modules so the other code revision can be imported fresh."""
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]


def build_old_model(checkpoint: Path, device: str):
    """Import the pre-fix models package from the baseline worktree and load its weights."""
    if not BASELINE_CODE.exists():
        raise FileNotFoundError(
            f"Baseline worktree missing at {BASELINE_CODE}. "
            "Recreate with: git worktree add _archive/baseline_code ad0f4d8"
        )
    _purge_project_modules()
    sys.path.insert(0, str(BASELINE_CODE))
    try:
        model_mod = importlib.import_module("models.model")
        # The pre-fix default was num_heads=1, which is what every eval script used.
        model = model_mod.PFGTUIEModel().to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        return model, state.get("epoch", -1), state.get("metrics", {})
    finally:
        sys.path.remove(str(BASELINE_CODE))


def build_new_model(checkpoint: Path, device: str):
    _purge_project_modules()
    from models.build import build_model
    from utils.checkpoint import load_checkpoint
    model = build_model(device=device)
    state = load_checkpoint(checkpoint, model=model, device=torch.device(device))
    model.eval()
    return model, state.get("epoch", -1), state.get("metrics", {})


def label(img: Image.Image, text: str) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 22), (18, 18, 22))
    out.paste(img, (0, 22))
    ImageDraw.Draw(out).text((5, 6), text, fill=(235, 235, 240))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--old-checkpoint", default="checkpoints/_baseline_before_fixes/best.pt")
    ap.add_argument("--out-dir", default="outputs/_phase1_check")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Sample from the HELD-OUT validation split only.
    from data.dataset import get_splits, subset_pair_names
    _, val_subset = get_splits(augment_train=False)
    names = subset_pair_names(val_subset)[: args.n]

    raw_dir = ROOT / "datasets" / "UIEB" / "raw-890"
    ref_dir = ROOT / "datasets" / "UIEB" / "reference-890"
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device

    old_model, old_epoch, old_metrics = build_old_model(Path(args.old_checkpoint), device)
    print(f"OLD model: {args.old_checkpoint} (epoch {old_epoch}, saved metrics {old_metrics.get('psnr', 'n/a')})")
    old_results = {}
    for n in names:
        raw = load_rgb(raw_dir / n).to(device)
        with torch.no_grad():
            old_results[n] = old_model(raw).cpu()
    del old_model
    torch.cuda.empty_cache()

    new_model, new_epoch, new_metrics = build_new_model(Path(args.new_checkpoint), device)
    print(f"NEW model: {args.new_checkpoint} (epoch {new_epoch}, saved metrics {new_metrics.get('psnr', 'n/a')})")
    new_results = {}
    for n in names:
        raw = load_rgb(raw_dir / n).to(device)
        with torch.no_grad():
            new_results[n] = new_model(raw).cpu()

    print()
    hdr = f"{'image':<18} {'OLD PSNR(raw,enh)':>18} {'NEW PSNR(raw,enh)':>18} {'OLD PSNR(ref,enh)':>18} {'NEW PSNR(ref,enh)':>18}"
    print(hdr)
    print("-" * len(hdr))

    sums = dict(old_id=0.0, new_id=0.0, old_ref=0.0, new_ref=0.0)
    for n in names:
        raw = load_rgb(raw_dir / n)
        ref = load_rgb(ref_dir / n)
        o, w = old_results[n], new_results[n]
        o_id, w_id = psnr(raw, o), psnr(raw, w)
        o_ref, w_ref = psnr(ref, o), psnr(ref, w)
        sums["old_id"] += o_id; sums["new_id"] += w_id
        sums["old_ref"] += o_ref; sums["new_ref"] += w_ref
        print(f"{n:<18} {o_id:>18.2f} {w_id:>18.2f} {o_ref:>18.2f} {w_ref:>18.2f}")

        panels = [
            label(to_pil(raw), "raw input (degraded)"),
            label(to_pil(o), f"OLD model  PSNR(raw)={o_id:.1f}"),
            label(to_pil(w), f"NEW model  PSNR(raw)={w_id:.1f}"),
            label(to_pil(ref), "reference (ground truth)"),
        ]
        W = sum(p.width for p in panels)
        grid = Image.new("RGB", (W, panels[0].height), (18, 18, 22))
        x = 0
        for p in panels:
            grid.paste(p, (x, 0)); x += p.width
        grid.save(out_dir / f"compare_{Path(n).stem}.png")

    k = len(names)
    print("-" * len(hdr))
    print(f"{'MEAN':<18} {sums['old_id']/k:>18.2f} {sums['new_id']/k:>18.2f} {sums['old_ref']/k:>18.2f} {sums['new_ref']/k:>18.2f}")
    print()
    print("Reading the table:")
    print("  PSNR(raw, enhanced) HIGH  => output is nearly the same image as the degraded input.")
    print("  PSNR(ref, enhanced) HIGH  => output is close to the ground-truth reference.")
    print(f"  OLD model sat {sums['old_id']/k:.1f} dB from its own input (near-copy).")
    print(f"  NEW model sits {sums['new_id']/k:.1f} dB from its input, i.e. it actually changes the image.")
    print(f"\nSide-by-side grids saved to: {out_dir}")


if __name__ == "__main__":
    main()
