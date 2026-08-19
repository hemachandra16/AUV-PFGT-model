"""PFGT-UIE Test Script.

Runs inference on the UIEB dataset (or a custom folder of paired images) and
saves enhanced outputs. Also computes PSNR, SSIM, UIQM, UCIQE when ground-
truth reference images are available.

Modes:
  --mode single   : enhance one image
  --mode folder   : enhance all images in a folder
  --mode dataset  : run on the full UIEB dataset

Usage:
    python test.py --mode single  --input path/to/raw.png
    python test.py --mode folder  --input path/to/raw_folder/
    python test.py --mode dataset --checkpoint checkpoints/best.pt
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

from data.dataset import get_splits, subset_pair_names
from models.build import build_model
from utils.checkpoint import load_checkpoint
from utils.logging_utils import setup_logger
from utils.seed import seed_everything

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test PFGT-UIE on images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to checkpoint file")
    parser.add_argument("--mode", type=str, default="dataset",
                        choices=["single", "folder", "dataset"],
                        help="Inference mode")
    parser.add_argument("--input", type=str, default=None,
                        help="Input image or folder path (for single/folder modes)")
    parser.add_argument("--raw-dir", type=str, default=None,
                        help="Raw images directory (dataset mode)")
    parser.add_argument("--reference-dir", type=str, default=None,
                        help="Reference images directory for metric computation (dataset mode)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory to save enhanced images")
    parser.add_argument("--image-size", type=int, default=256,
                        help="Resize target for inference")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "full"],
                        help="Dataset-mode split. 'val' (default) = held-out 10%%; "
                             "'train'/'full' overlap training data, diagnostics only")
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_image(path: Path, image_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Load an image and return as (1, 3, H, W) float tensor in [0,1] and original size."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        orig_size = img.size
        img = img.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, orig_size


def save_image(tensor: torch.Tensor, path: Path) -> None:
    """Save a (1, 3, H, W) or (3, H, W) float tensor in [0,1] as a PNG."""
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    arr = (tensor.detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    Image.fromarray(arr).save(path)


@torch.no_grad()
def enhance_image(
    model: torch.nn.Module,
    image_path: Path,
    output_dir: Path,
    device: torch.device,
    image_size: int,
) -> Path:
    """Enhance a single image and save the result."""
    tensor, orig_size = load_image(image_path, image_size)
    tensor = tensor.to(device)
    output = model(tensor)
    output_resized = torch.nn.functional.interpolate(
        output, size=(orig_size[1], orig_size[0]), mode="bilinear", align_corners=False
    )
    out_path = output_dir / image_path.name
    save_image(output_resized, out_path)
    return out_path


def collect_images(folder: Path) -> list[Path]:
    """Return all image files in a folder."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def main() -> None:
    args = parse_args()
    setup_logger(log_dir="logs", log_file="test.log")
    seed_everything(args.seed)

    device = get_device(args.device)
    logger.info("Test device: %s", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = build_model(device=device)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    if args.mode == "single":
        if args.input is None:
            raise ValueError("--input must be specified in single mode.")
        path = Path(args.input)
        out = enhance_image(model, path, output_dir, device, args.image_size)
        logger.info("Enhanced image saved to: %s", out)

    elif args.mode == "folder":
        if args.input is None:
            raise ValueError("--input must be specified in folder mode.")
        images = collect_images(Path(args.input))
        logger.info("Found %d images in %s", len(images), args.input)
        for img_path in images:
            out = enhance_image(model, img_path, output_dir, device, args.image_size)
            logger.info("  Saved: %s", out)

    elif args.mode == "dataset":
        # Auto-detect UIEB dataset paths
        root = Path(__file__).resolve().parent
        raw_dir = Path(args.raw_dir) if args.raw_dir else root / "datasets" / "UIEB" / "raw-890"
        ref_dir = Path(args.reference_dir) if args.reference_dir else root / "datasets" / "UIEB" / "reference-890"

        # Use the SAME seeded split as train.py so reported metrics are genuinely
        # held-out. Scoring all 890 pairs (the old behaviour) included the ~801 training
        # images and inflated the numbers in logs/test.log.
        train_subset, val_subset = get_splits(
            raw_dir=args.raw_dir,
            reference_dir=args.reference_dir,
            image_size=args.image_size,
            augment_train=False,
        )
        if args.split == "val":
            names = subset_pair_names(val_subset)
        elif args.split == "train":
            names = subset_pair_names(train_subset)
        else:
            names = [p.name for p in collect_images(raw_dir)]
            logger.warning(
                "--split full scores TRAINING IMAGES TOO; not a held-out result."
            )
        images = [raw_dir / n for n in names]
        logger.info("Dataset mode (split=%s): %d images from %s", args.split, len(images), raw_dir)

        psnr_sum = ssim_sum = 0.0
        n = 0
        has_metrics = ref_dir.exists()

        if has_metrics:
            from metrics import compute_psnr, compute_ssim

        for img_path in images:
            tensor, orig_size = load_image(img_path, args.image_size)
            tensor = tensor.to(device)
            with torch.no_grad():
                output = model(tensor)
            output_resized = torch.nn.functional.interpolate(
                output, size=(orig_size[1], orig_size[0]), mode="bilinear", align_corners=False
            )
            out_path = output_dir / img_path.name
            save_image(output_resized, out_path)

            if has_metrics and (ref_dir / img_path.name).exists():
                ref_tensor, _ = load_image(ref_dir / img_path.name, args.image_size)
                ref_tensor = ref_tensor.to(device)
                psnr_sum += compute_psnr(output, ref_tensor)
                ssim_sum += compute_ssim(output, ref_tensor)
                n += 1

        logger.info("Saved %d enhanced images to: %s", len(images), output_dir)
        if n > 0:
            logger.info("Split=%s | Average PSNR: %.4f dB | Average SSIM: %.4f",
                        args.split, psnr_sum / n, ssim_sum / n)

    logger.info("Test complete.")


if __name__ == "__main__":
    main()
