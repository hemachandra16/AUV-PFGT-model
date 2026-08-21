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
from models.object_detection import build_detector, annotate_image_with_detections
from utils.checkpoint import load_checkpoint
from utils.logging_utils import setup_logger

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Underwater object detection, with optional PFGT-UIE enhancement",
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
    parser.add_argument("--conf-thresh", type=float, default=0.25,
                        help="Detection confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.45,
                        help="NMS IoU threshold (now actually applied)")
    parser.add_argument("--detector", type=str, default="auto",
                        choices=["auto", "yolo", "fasterrcnn"],
                        help="'yolo' = RUOD fine-tuned (real underwater classes); "
                             "'fasterrcnn' = legacy COCO model with hand-mapped names; "
                             "'auto' = yolo if weights exist, else fasterrcnn")
    parser.add_argument("--detector-weights", type=str, default=None,
                        help="Path to fine-tuned detector weights")
    parser.add_argument("--detect-on", type=str, default="raw",
                        choices=["raw", "enhanced"],
                        help="Which image the DETECTOR sees. Default 'raw': session 2's "
                             "ablation measured that detecting on enhanced frames costs "
                             "3.9 mAP@0.5 (0.8292 -> 0.7906), because the detector was "
                             "fine-tuned on raw RUOD frames. Use 'enhanced' only if you "
                             "have retrained the detector on enhanced imagery.")
    parser.add_argument("--annotate-on", type=str, default="enhanced",
                        choices=["raw", "enhanced"],
                        help="Which image the boxes are drawn on for viewing. Enhanced by "
                             "default, since enhancement helps a human read the scene even "
                             "though it hurts the detector.")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Skip enhancement entirely (fastest; annotates the raw frame)")
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
    enhancer,
    detector,
    image_path: Path,
    output_path: Path,
    device: torch.device,
    conf_thresh: float,
    iou_thresh: float = 0.45,
    detect_on: str = "raw",
    annotate_on: str = "enhanced",
) -> None:
    """Detect on one image, optionally enhancing first.

    The detector and the human do not want the same picture. Session 2's ablation
    (results/ablation_enhance_detect.json) measured that feeding PFGT-UIE-enhanced frames
    to the RUOD-fine-tuned detector costs 3.9 mAP@0.5 (0.8292 -> 0.7906), because the
    detector was trained on raw frames and enhancement shifts the input distribution
    (it raises contrast ~40%, so it is a domain shift, not a loss of detail).

    So by default the detector sees the RAW frame and the boxes are drawn on the ENHANCED
    frame — full detection accuracy, and a picture a person can actually read.
    """
    # 1. Load image at full native resolution
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        orig_w, orig_h = img.size
        arr = np.array(img, dtype=np.float32) / 255.0

    raw_tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    enhanced_tensor = None
    if enhancer is not None:
        # 2. Pad H, W to multiples of 16 for the wavelet transform
        pad_h = (16 - orig_h % 16) % 16
        pad_w = (16 - orig_w % 16) % 16
        padded = F.pad(raw_tensor, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else raw_tensor
        with torch.no_grad():
            out_padded = enhancer(padded)
        enhanced_tensor = out_padded[:, :, :orig_h, :orig_w]

    # 3. Detect on whichever frame was requested (raw by default)
    if detect_on == "enhanced" and enhanced_tensor is not None:
        detect_tensor = enhanced_tensor
    else:
        detect_tensor = raw_tensor
        if detect_on == "enhanced":
            logger.warning("--detect-on enhanced requested but enhancement is disabled; "
                           "detecting on the raw frame.")

    detections = detector.detect_objects(detect_tensor, conf_threshold=conf_thresh,
                                         iou_threshold=iou_thresh)

    # 4. Draw the boxes on whichever frame reads best for a human
    if annotate_on == "enhanced" and enhanced_tensor is not None:
        canvas_tensor = enhanced_tensor
    else:
        canvas_tensor = raw_tensor
    canvas_arr = (canvas_tensor.squeeze(0).detach().cpu().clamp(0, 1)
                  .numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
    canvas_pil = Image.fromarray(canvas_arr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotate_image_with_detections(canvas_pil, detections, output_path=output_path)
    logger.info("Saved %s (%dx%d) | detected on %s, annotated on %s | %d objects",
                output_path, orig_w, orig_h, detect_on,
                annotate_on if enhanced_tensor is not None else "raw", len(detections))


def main() -> None:
    args = parse_args()
    setup_logger(log_dir="logs", log_file="infer_detection.log")

    device = get_device(args.device)
    logger.info("Running native detection inference on device: %s", device)

    # Load the enhancement model only if it will actually be used.
    enhancer = None
    if not args.no_enhance:
        enhancer = build_model(device=device)
        load_checkpoint(args.checkpoint, model=enhancer, device=device)
        enhancer.eval()
    logger.info("Detector sees: %s frames | annotation on: %s",
                args.detect_on,
                "raw" if (args.no_enhance or args.annotate_on == "raw") else "enhanced")
    if args.detect_on == "raw":
        logger.info("(default) detecting on raw frames — enhancement costs 3.9 mAP@0.5 "
                    "for this detector; see results/ablation_enhance_detect.json")

    # Load Detector Model
    detector = build_detector(
        backend=args.detector,
        weights=args.detector_weights,
        conf_threshold=args.conf_thresh,
        iou_threshold=args.iou_thresh,
    )
    detector.eval()
    if hasattr(detector, "model") and hasattr(detector.model, "to"):
        try:
            detector.to(device)
        except Exception:
            pass  # Ultralytics manages its own device placement.

    input_path = Path(args.input)

    if input_path.is_file():
        out_path = Path(args.output) if args.output else Path(args.output_dir) / f"detected_{input_path.name}"
        process_image(enhancer, detector, input_path, out_path, device, args.conf_thresh,
                      args.iou_thresh, args.detect_on, args.annotate_on)

    elif input_path.is_dir():
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        images = sorted(p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS)
        logger.info("Found %d images in %s", len(images), input_path)

        for img_path in images:
            out_path = out_dir / f"detected_{img_path.name}"
            process_image(enhancer, detector, img_path, out_path, device, args.conf_thresh,
                          args.iou_thresh, args.detect_on, args.annotate_on)

        logger.info("Processed %d images into %s", len(images), out_dir)


if __name__ == "__main__":
    main()
