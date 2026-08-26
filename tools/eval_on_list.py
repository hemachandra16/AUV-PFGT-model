"""Score a checkpoint on an explicit list of UIEB filenames, partitioned by contamination.

`validate.py` can only score this project's own seeded split. Comparing against a published
test set needs an arbitrary filename list, and — more importantly — needs to be honest about
the fact that most of that list may already be in the model's training data.

So this scores three groups separately and labels them:

  ALL      every name in the list. Meaningless as a test result if any of it was trained on.
  SEEN     the subset this model was trained on. Reported only to quantify how much
           memorisation inflates the ALL number.
  UNSEEN   the subset genuinely held out from this model. The only honest figure here, and
           only as strong as its sample size allows.

Metric computation is delegated to `validate.validate` and the model is built and loaded
exactly as `validate.py` does it, so the numbers are directly comparable to every figure this
project has previously reported.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import Subset

from data.dataset import UIEBDataset, get_splits, subset_pair_names
from models.build import build_model
from utils.checkpoint import load_checkpoint
from utils.logging_utils import setup_logger
from utils.seed import seed_everything
from validate import get_device, validate

logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--list", required=True, help="text file, one image filename per line")
    ap.add_argument("--label", default="list", help="name used in output paths and logs")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    setup_logger(log_dir="logs", log_file="eval_on_list.log")
    seed_everything(args.seed)
    device = get_device(args.device)

    wanted = [l.strip() for l in Path(args.list).read_text(encoding="utf-8").splitlines()
              if l.strip()]
    logger.info("Requested list '%s': %d filenames", args.label, len(wanted))

    # Same split call every other script uses, so "trained on" means exactly what it means
    # everywhere else in this project.
    train_subset, val_subset = get_splits(image_size=args.image_size, augment_train=False)
    trained_on = set(subset_pair_names(train_subset))
    held_out = set(subset_pair_names(val_subset))

    full = UIEBDataset(image_size=args.image_size)
    index_of = {p[0].name: i for i, p in enumerate(full.pairs)}

    missing = [n for n in wanted if n not in index_of]
    if missing:
        logger.error("%d requested names are not present locally: %s", len(missing), missing[:8])
        raise SystemExit(1)

    groups = {
        "ALL": wanted,
        "SEEN-in-training": [n for n in wanted if n in trained_on],
        "UNSEEN-by-this-model": [n for n in wanted if n not in trained_on],
    }
    sanity = [n for n in wanted if n not in trained_on and n not in held_out]
    assert not sanity, f"names in neither split: {sanity[:5]}"

    model = build_model(device=device)
    load_checkpoint(args.checkpoint, model=model, device=device)
    logger.info("Checkpoint loaded: %s", args.checkpoint)

    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    for gname, names in groups.items():
        if not names:
            logger.warning("group %s is empty, skipping", gname)
            continue
        subset = Subset(full, [index_of[n] for n in names])
        csv_path = os.path.join(args.out_dir, f"{args.label}_{gname}.csv")
        logger.info("")
        logger.info("--- %s : %d images ---", gname, len(names))
        results[gname] = validate(model=model, dataset=subset, pair_names=names,
                                  batch_size=args.batch_size, device=device,
                                  output_csv=csv_path)

    print("\n" + "=" * 76)
    print(f"  {args.label}  |  checkpoint: {args.checkpoint}")
    print("=" * 76)
    print(f"  {'group':<24}{'n':>5}{'PSNR':>10}{'SSIM':>9}{'UIQM':>9}{'UCIQE':>9}")
    for gname, m in results.items():
        print(f"  {gname:<24}{m['n_images']:>5}{m['psnr']:>10.4f}{m['ssim']:>9.4f}"
              f"{m['uiqm']:>9.4f}{m['uciqe']:>9.4f}")
    print("=" * 76)

    seen = results.get("SEEN-in-training")
    unseen = results.get("UNSEEN-by-this-model")
    if seen and unseen:
        print(f"  memorisation gap (SEEN - UNSEEN): "
              f"{seen['psnr'] - unseen['psnr']:+.3f} dB PSNR, "
              f"{seen['ssim'] - unseen['ssim']:+.4f} SSIM")
        print(f"  the ALL row is {100*seen['n_images']/len(wanted):.0f}% training data and is "
              f"NOT a valid test result.")
    print()


if __name__ == "__main__":
    main()
