"""Evaluate the PRE-FIX baseline checkpoint honestly, for a fair comparison.

Three things this settles:

1. **What the baseline actually scores on held-out data.** The old ``validate.py`` and
   ``test.py --mode dataset`` scored all 890 UIEB pairs, ~801 of which the model had
   trained on. This re-scores the same checkpoint on the seeded held-out 89 only.

2. **What it scores with a correct UCIQE.** ``metrics/uciqe.py`` was broken (see
   PROGRESS.md D-010); every previously reported UCIQE is meaningless. Both models are
   scored here with the fixed metric.

3. **How much the silent ``num_heads`` mismatch cost.** train.py built the model with
   ``num_heads=4`` from the config; every evaluation script used the class default
   ``num_heads=1``. No weight shape depends on num_heads, so the checkpoint loaded
   cleanly and ran at the wrong attention temperature. Scoring the same weights both
   ways measures the damage directly.

The pre-fix model code is loaded from the git worktree at ``_archive/baseline_code``
(commit ad0f4d8), because the fixed attention module has parameters the old checkpoint
does not contain.

Usage:
    python tools/eval_baseline.py --device cpu
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASELINE_CODE = ROOT / "_archive" / "baseline_code"


def purge_project_modules() -> None:
    for name in list(sys.modules):
        if name == "models" or name.startswith("models."):
            del sys.modules[name]


def load_old_model(num_heads: int, checkpoint: Path, device: str):
    """Instantiate the PRE-FIX architecture with a given head count and load weights."""
    purge_project_modules()
    sys.path.insert(0, str(BASELINE_CODE))
    try:
        model_mod = importlib.import_module("models.model")
        model = model_mod.PFGTUIEModel(embed_dim=128, num_heads=num_heads).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        model.eval()
        return model, state
    finally:
        sys.path.remove(str(BASELINE_CODE))


@torch.no_grad()
def score(model, subset, device: str, batch_size: int = 4) -> dict:
    from metrics import compute_psnr, compute_ssim, compute_uiqm, compute_uciqe

    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    sums = {"psnr": 0.0, "ssim": 0.0, "uiqm": 0.0, "uciqe": 0.0}
    n = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        for i in range(outputs.shape[0]):
            p, t = outputs[i:i + 1], targets[i:i + 1]
            sums["psnr"] += compute_psnr(p, t)
            sums["ssim"] += compute_ssim(p, t)
            sums["uiqm"] += compute_uiqm(p)
            sums["uciqe"] += compute_uciqe(p)
            n += 1
    return {k: v / max(n, 1) for k, v in sums.items()} | {"n_images": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/_baseline_before_fixes/best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--out", default="results/baseline_metrics.json")
    args = ap.parse_args()

    from data.dataset import get_splits

    train_subset, val_subset = get_splits(augment_train=False)
    print(f"Splits: {len(train_subset)} train / {len(val_subset)} held-out val")

    ckpt_path = Path(args.checkpoint)
    results: dict = {
        "checkpoint": str(ckpt_path),
        "note": "Pre-fix architecture (commit ad0f4d8) scored with the FIXED UCIQE metric.",
    }

    for heads in (1, 4):
        model, state = load_old_model(heads, ckpt_path, args.device)
        results.setdefault("checkpoint_epoch", state.get("epoch"))
        results.setdefault("checkpoint_saved_metrics", state.get("metrics", {}))

        print(f"\n--- baseline, num_heads={heads} ---")
        held_out = score(model, val_subset, args.device, args.batch_size)
        print(f"  HELD-OUT val ({held_out['n_images']} imgs): "
              f"PSNR {held_out['psnr']:.4f}  SSIM {held_out['ssim']:.4f}  "
              f"UIQM {held_out['uiqm']:.4f}  UCIQE {held_out['uciqe']:.4f}")
        results[f"heads_{heads}_val"] = held_out

        # Only for heads=1 (what the old scripts actually used) also score the training
        # split, to quantify the leakage that inflated the old reported numbers.
        if heads == 1:
            train_scored = score(model, train_subset, args.device, args.batch_size)
            print(f"  TRAIN split ({train_scored['n_images']} imgs): "
                  f"PSNR {train_scored['psnr']:.4f}  SSIM {train_scored['ssim']:.4f}")
            results["heads_1_train"] = train_scored
            n_v, n_t = held_out["n_images"], train_scored["n_images"]
            results["heads_1_full_890_estimate"] = {
                "psnr": (held_out["psnr"] * n_v + train_scored["psnr"] * n_t) / (n_v + n_t),
                "ssim": (held_out["ssim"] * n_v + train_scored["ssim"] * n_t) / (n_v + n_t),
                "note": "What the old leaky evaluation would have reported over all 890.",
            }
        del model

    h1 = results["heads_1_val"]["psnr"]
    h4 = results["heads_4_val"]["psnr"]
    results["num_heads_mismatch_psnr_delta"] = h4 - h1

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("BASELINE (pre-fix checkpoint), scored on the held-out 89 with fixed UCIQE")
    print("=" * 72)
    print(f"  as evaluated by the old scripts (num_heads=1): PSNR {h1:.4f} dB")
    print(f"  as actually trained          (num_heads=4): PSNR {h4:.4f} dB")
    print(f"  cost of the silent mismatch              : {h4 - h1:+.4f} dB")
    if "heads_1_full_890_estimate" in results:
        est = results["heads_1_full_890_estimate"]
        print(f"\n  old leaky all-890 evaluation would report : PSNR {est['psnr']:.4f} dB")
        print(f"  honest held-out figure                    : PSNR {h1:.4f} dB")
        print(f"  inflation from train/eval leakage         : {est['psnr'] - h1:+.4f} dB")
    print("=" * 72)
    print(f"saved -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
