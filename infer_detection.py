"""PFGT-UIE Combined Native Resolution Enhancement & Object Detection Script.

Enhances raw underwater images at full native resolution using PFGT-UIE and
detects marine life/underwater objects with a pretrained Faster R-CNN detector.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.build import build_model
from models.object_detection import UnderwaterObjectDetector, annotate_image_with_detections
from utils.checkpoint import load_checkpoint
from utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PFGT-UIE Native Enhancement + Pretrained Object Detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to PFGT-UIE checkpoint")
    parser.add_argument("--input", type=str, required=True,
                        help="Input image file or folder of images")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for single image mode")
    parser.add_argument("--output-dir", type=str, default="outputs/detection",
                        help="Output directory for folder mode")
    parser.add_argument("--conf-thresh", type=float, default=0.45,
                        help="Detection confidence threshold")
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


def process_image(
    enhancer: torch.nn.Module,
    detector: UnderwaterObjectDetector,
    image_path: Path,
    output_path: Path,
    device: torch.device,
    conf_thresh: float,
) -> None:
    # 1. Load Image at full native resolution
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        arr = np.array(img, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    # 2. Pad H, W to multiples of 16 for model processing
    pad_h = (16 - orig_h % 16) % 16
    pad_w = (16 - orig_w % 16) % 16

    if pad_h > 0 or pad_w > 0:
        tensor_padded = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    else:
        tensor_padded = tensor

    # 3. Enhance image with PFGT-UIE
    with torch.no_grad():
        output_padded = enhancer(tensor_padded)

    # Crop back to exact native resolution
    enhanced_tensor = output_padded[:, :, :orig_h, :orig_w]

    # 4. Detect objects on native-resolution enhanced image
    detections = detector.detect_objects(enhanced_tensor, conf_threshold=conf_thresh)

    # 5. Convert enhanced tensor to PIL Image & annotate
    enhanced_arr = (enhanced_tensor.squeeze(0).detach().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    enhanced_pil = Image.fromarray(enhanced_arr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotate_image_with_detections(enhanced_pil, detections, output_path=output_path)
    logger.info("Saved native-res enhanced + detected output to: %s (%dx%d, %d objects detected)", output_path, orig_w, orig_h, len(detections))


def main() -> None:
    args = parse_args()
    setup_logger(log_dir="logs", log_file="infer_detection.log")

    device = get_device(args.device)
    logger.info("Running native detection inference on device: %s", device)

    # Load Enhancement Model
    enhancer = build_model(device=device)
    load_checkpoint(args.checkpoint, model=enhancer, device=device)
    enhancer.eval()

    # Load Detector Model
    detector = UnderwaterObjectDetector(conf_threshold=args.conf_thresh).to(device)
    detector.eval()

    input_path = Path(args.input)

    if input_path.is_file():
        out_path = Path(args.output) if args.output else Path(args.output_dir) / f"detected_{input_path.name}"
        process_image(enhancer, detector, input_path, out_path, device, args.conf_thresh)

    elif input_path.is_dir():
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
        logger.info("Found %d images in %s", len(images), input_path)

        for img_path in images:
            out_path = out_dir / f"detected_{img_path.name}"
            process_image(enhancer, detector, img_path, out_path, device, args.conf_thresh)

        logger.info("Processed %d images into %s", len(images), out_dir)


if __name__ == "__main__":
    main()
