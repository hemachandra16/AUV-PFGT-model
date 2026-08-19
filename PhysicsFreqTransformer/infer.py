"""PFGT-UIE Native Resolution Inference Script.

Processes underwater images directly at native resolution with zero downsampling.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.model import PFGTUIEModel
from utils.checkpoint import load_checkpoint
from utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native-resolution inference with PFGT-UIE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to .pt checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image file or folder of images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for single image mode")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Output directory for folder mode")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def process_native_image(model: torch.nn.Module, image_path: Path, device: torch.device) -> Image.Image:
    """Process image at full native resolution with padding to multiple of 16."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        arr = np.array(img, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # Pad H, W to multiples of 16 for wavelet transform
    pad_h = (16 - orig_h % 16) % 16
    pad_w = (16 - orig_w % 16) % 16

    if pad_h > 0 or pad_w > 0:
        tensor_padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        tensor_padded = tensor

    with torch.no_grad():
        output_padded = model(tensor_padded)

    # Crop output back to exact original native dimensions
    output = output_padded[:, :, :orig_h, :orig_w]
    arr_out = (output.squeeze(0).detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr_out)


def main() -> None:
    args = parse_args()
    setup_logger(log_dir="logs", log_file="infer.log")

    device = get_device(args.device)
    logger.info("Native resolution inference on device: %s", device)

    model = PFGTUIEModel().to(device)
    load_checkpoint(args.checkpoint, model=model, device=device)
    model.eval()
    logger.info("Loaded checkpoint: %s", args.checkpoint)

    input_path = Path(args.input)

    if input_path.is_file():
        enhanced_img = process_native_image(model, input_path, device)
        out_path = Path(args.output) if args.output else Path(args.output_dir) / input_path.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        enhanced_img.save(out_path)
        logger.info("Saved native-resolution enhanced image to: %s (%dx%d)", out_path, enhanced_img.width, enhanced_img.height)

    elif input_path.is_dir():
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
        logger.info("Found %d images in %s", len(images), input_path)

        for img_path in images:
            enhanced_img = process_native_image(model, img_path, device)
            out_path = out_dir / img_path.name
            enhanced_img.save(out_path)

        logger.info("Saved %d native-resolution enhanced images to: %s", len(images), out_dir)


if __name__ == "__main__":
    main()
