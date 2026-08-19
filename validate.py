"""PFGT-UIE Validation Script.

Evaluates a trained checkpoint on the UIEB validation or test split.
Reports PSNR, SSIM, UIQM, and UCIQE per image and as averages.
Saves a per-image CSV report to the output directory.

Usage:
    python validate.py --checkpoint checkpoints/best.pt
    python validate.py --checkpoint checkpoints/best.pt --split full
    python validate.py --checkpoint checkpoints/best.pt --output-csv results/metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import UIEBDataset
from metrics import compute_psnr, compute_ssim, compute_uiqm, compute_uciqe
from models.model import PFGTUIEModel
from utils.checkpoint import load_checkpoint
from utils.logging_utils import setup_logger
from utils.seed import seed_everything

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a PFGT-UIE checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best.pt",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
        help="Device to run validation on",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Validation batch size",
    )
    parser.add_argument(
        "--image-size", type=int, default=256,
        help="Image resize target",
    )
    parser.add_argument(
        "--output-csv", type=str, default="results/validation_metrics.csv",
        help="Path to save per-image CSV report",
    )
    parser.add_argument(
        "--raw-dir", type=str, default=None,
        help="Raw images directory (auto-detected if None)",
    )
    parser.add_argument(
        "--reference-dir", type=str, default=None,
        help="Reference images directory (auto-detected if None)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataset: UIEBDataset,
    batch_size: int,
    device: torch.device,
    output_csv: str,
) -> dict[str, float]:
    """Run full validation and return averaged metrics."""
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    rows: list[dict] = []
    psnr_sum = 0.0
    ssim_sum = 0.0
    uiqm_sum = 0.0
    uciqe_sum = 0.0
    n_images = 0

    pair_names = [p[0].name for p in dataset.pairs]

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)

        # Per-image metrics in the batch
        for i in range(outputs.shape[0]):
            pred_i = outputs[i : i + 1]
            tgt_i = targets[i : i + 1]

            psnr = compute_psnr(pred_i, tgt_i)
            ssim = compute_ssim(pred_i, tgt_i)
            uiqm = compute_uiqm(pred_i)
            uciqe = compute_uciqe(pred_i)

            img_idx = batch_idx * batch_size + i
            name = pair_names[img_idx] if img_idx < len(pair_names) else f"image_{img_idx}"

            rows.append({
                "filename": name,
                "psnr": f"{psnr:.4f}",
                "ssim": f"{ssim:.4f}",
                "uiqm": f"{uiqm:.4f}",
                "uciqe": f"{uciqe:.4f}",
            })

            psnr_sum += psnr
            ssim_sum += ssim
            uiqm_sum += uiqm
            uciqe_sum += uciqe
            n_images += 1

    n = max(n_images, 1)
    avg = {
        "psnr": psnr_sum / n,
        "ssim": ssim_sum / n,
        "uiqm": uiqm_sum / n,
        "uciqe": uciqe_sum / n,
        "n_images": n_images,
    }

    # Save CSV report
    output_csv_path = Path(output_csv)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "psnr", "ssim", "uiqm", "uciqe"])
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Per-image metrics saved to: %s", output_csv_path)

    return avg


def main() -> None:
    args = parse_args()
    setup_logger(log_dir="logs", log_file="validate.log")
    seed_everything(args.seed)

    device = get_device(args.device)
    logger.info("Validating on device: %s", device)

    # Load dataset
    dataset = UIEBDataset(
        raw_dir=args.raw_dir,
        reference_dir=args.reference_dir,
        image_size=args.image_size,
    )
    logger.info("Dataset: %d paired images", len(dataset))

    # Load model
    model = PFGTUIEModel().to(device)
    load_checkpoint(args.checkpoint, model=model, device=device)
    logger.info("Checkpoint loaded: %s", args.checkpoint)

    # Run validation
    logger.info("Running validation on %d images...", len(dataset))
    metrics = validate(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        device=device,
        output_csv=args.output_csv,
    )

    # Report
    logger.info("")
    logger.info("=" * 50)
    logger.info("  VALIDATION RESULTS  (%d images)", metrics["n_images"])
    logger.info("=" * 50)
    logger.info("  PSNR  : %.4f dB", metrics["psnr"])
    logger.info("  SSIM  : %.4f", metrics["ssim"])
    logger.info("  UIQM  : %.4f", metrics["uiqm"])
    logger.info("  UCIQE : %.4f", metrics["uciqe"])
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
