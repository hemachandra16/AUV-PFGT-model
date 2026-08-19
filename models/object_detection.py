"""Underwater object detection for PFGT-UIE.

Two backends are available:

``UnderwaterYOLODetector`` (DEFAULT)
    A YOLO model fine-tuned on RUOD (Real-world Underwater Object Detection): 14,000
    real underwater images, 10 marine classes. Trained by ``tools/train_detector.py``.
    Class names are the dataset's own, so a "starfish" box means a starfish was detected.

``COCOFasterRCNNDetector`` (LEGACY, kept for comparison only)
    The previous approach: torchvision's Faster R-CNN pretrained on COCO — a dataset of
    everyday above-water photographs — with its predictions renamed through a hand-written
    dictionary (``"frisbee" -> "starfish"``, ``"kite" -> "stingray"``, ``"bear" ->
    "marine_life"``, and so on). It had never seen an underwater image. The renaming does
    not add underwater knowledge; it only relabels whatever COCO object the network
    happened to fire on, so a genuine starfish is only ever called one if it first looks
    like a frisbee. It is retained so the fine-tuned model can be compared against it, and
    as a fallback when no fine-tuned weights are present.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

# RUOD's 10 classes, in dataset order.
RUOD_CLASSES = [
    "holothurian", "echinus", "scallop", "starfish", "fish",
    "corals", "diver", "cuttlefish", "turtle", "jellyfish",
]

DEFAULT_YOLO_WEIGHTS = Path(__file__).resolve().parents[1] / "checkpoints" / "detector" / "best.pt"

CLASS_COLORS = {
    "holothurian": (255, 140, 0),     # dark orange
    "echinus": (147, 112, 219),       # medium purple
    "scallop": (255, 215, 0),         # gold
    "starfish": (255, 99, 71),        # tomato
    "fish": (0, 255, 127),            # spring green
    "corals": (255, 105, 180),        # hot pink
    "diver": (30, 144, 255),          # dodger blue
    "cuttlefish": (0, 206, 209),      # dark turquoise
    "turtle": (154, 205, 50),         # yellow green
    "jellyfish": (238, 130, 238),     # violet
}
DEFAULT_COLOR = (0, 255, 127)


# ---------------------------------------------------------------------------
# Fine-tuned YOLO detector (default)
# ---------------------------------------------------------------------------

class UnderwaterYOLODetector(nn.Module):
    """Underwater detector backed by a YOLO model fine-tuned on RUOD."""

    def __init__(
        self,
        weights: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "ultralytics is required for the fine-tuned underwater detector. "
                "Install it with 'pip install ultralytics'."
            ) from exc

        weights_path = Path(weights) if weights is not None else DEFAULT_YOLO_WEIGHTS
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Fine-tuned detector weights not found at {weights_path}. "
                "Train them with: python tools/train_detector.py"
            )

        self.weights_path = weights_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.model = YOLO(str(weights_path))
        # Prefer the names baked into the checkpoint; fall back to the known RUOD order.
        names = getattr(self.model, "names", None)
        if isinstance(names, dict):
            self.class_names = [names[i] for i in sorted(names)]
        elif isinstance(names, (list, tuple)):
            self.class_names = list(names)
        else:
            self.class_names = list(RUOD_CLASSES)
        logger.info("Loaded fine-tuned underwater detector: %s (%d classes)",
                    weights_path, len(self.class_names))

    @torch.no_grad()
    def detect_objects(
        self,
        image_tensor: torch.Tensor,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Detect underwater objects in a (1, 3, H, W) or (3, H, W) float tensor in [0, 1].

        Unlike the legacy detector, ``iou_threshold`` is actually applied — it is passed
        to the model's NMS.
        """
        conf = self.conf_threshold if conf_threshold is None else conf_threshold
        iou = self.iou_threshold if iou_threshold is None else iou_threshold

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Ultralytics expects HWC uint8 RGB for a numpy source.
        arr = (image_tensor[0].detach().float().cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0)
        arr = arr.astype(np.uint8)

        device = 0 if torch.cuda.is_available() else "cpu"
        results = self.model.predict(
            source=arr[..., ::-1],  # ultralytics numpy sources are BGR
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None:
            return detections

        for box in boxes:
            cls_idx = int(box.cls.item())
            score = float(box.conf.item())
            x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
            name = self.class_names[cls_idx] if cls_idx < len(self.class_names) else f"class_{cls_idx}"
            detections.append({
                "class": name,
                "confidence": score,
                "bbox": [int(x0), int(y0), int(x1), int(y1)],
                "raw_class": name,
            })
        return detections


# ---------------------------------------------------------------------------
# Legacy COCO Faster R-CNN detector (comparison / fallback only)
# ---------------------------------------------------------------------------

# The hand-written COCO->underwater renaming used by the previous implementation.
# Retained verbatim so the legacy path still behaves as it did, and so the report can
# point at exactly what was being claimed as "underwater detection".
COCO_UNDERWATER_MAP = {
    "person": "diver", "boat": "underwater_vehicle", "bird": "sea_bird",
    "cat": "marine_life", "dog": "marine_life", "horse": "sea_horse",
    "sheep": "coral", "cow": "sea_cow", "elephant": "marine_life",
    "bear": "marine_life", "zebra": "fish", "giraffe": "coral",
    "backpack": "underwater_gear", "umbrella": "jellyfish", "handbag": "underwater_debris",
    "tie": "sea_weed", "suitcase": "underwater_equipment", "frisbee": "starfish",
    "skis": "diver_fins", "sports ball": "sea_urchin", "kite": "stingray",
    "baseball bat": "pipe", "bottle": "underwater_debris", "cup": "underwater_debris",
    "fork": "underwater_debris", "knife": "underwater_tools", "spoon": "underwater_debris",
    "bowl": "shell", "banana": "sea_cucumber", "apple": "sea_anemone",
    "sandwich": "sponge", "orange": "sea_urchin", "broccoli": "coral",
    "carrot": "coral", "hot dog": "sea_cucumber", "pizza": "starfish",
    "donut": "lifebuoy", "cake": "coral", "chair": "submerged_structure",
    "couch": "submerged_structure", "potted plant": "coral", "bed": "reef_structure",
    "dining table": "reef_structure", "toilet": "underwater_debris", "tv": "underwater_equipment",
    "laptop": "underwater_equipment", "mouse": "sensor", "remote": "sensor",
    "cell phone": "camera_gear", "sink": "underwater_debris", "refrigerator": "submerged_debris",
    "book": "underwater_slate", "clock": "gauge", "vase": "amphora",
    "scissors": "underwater_tools", "teddy bear": "marine_life", "hair drier": "underwater_tools",
    "toothbrush": "underwater_debris",
}


class COCOFasterRCNNDetector(nn.Module):
    """LEGACY: COCO-pretrained Faster R-CNN with hand-mapped class names.

    Never trained on underwater imagery. Kept only for comparison against the fine-tuned
    RUOD detector, and as a fallback when no fine-tuned weights exist.
    """

    def __init__(self, conf_threshold: float = 0.45, iou_threshold: float = 0.45) -> None:
        super().__init__()
        import torchvision.models.detection as detection

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        # box_nms_thresh wires iou_threshold to the model's actual NMS. In the previous
        # implementation this parameter was accepted and then never used.
        self.model = detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights, box_nms_thresh=iou_threshold
        )
        self.coco_categories = weights.meta["categories"]
        self.model.eval()
        logger.warning(
            "Using LEGACY COCO Faster R-CNN detector with hand-mapped class names. "
            "It has never seen underwater imagery; its labels are not trustworthy."
        )

    @torch.no_grad()
    def detect_objects(
        self,
        image_tensor: torch.Tensor,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        self.eval()
        thresh = self.conf_threshold if conf_threshold is None else conf_threshold

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        device = next(self.parameters()).device
        image_tensor = image_tensor.to(device)

        predictions = self.model(image_tensor)[0]
        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        detections: List[Dict[str, Any]] = []
        for box, score, label_id in zip(boxes, scores, labels):
            if score < thresh:
                continue
            raw_class = (self.coco_categories[label_id]
                         if label_id < len(self.coco_categories) else "object")
            mapped_class = COCO_UNDERWATER_MAP.get(raw_class, "marine_life")
            xmin, ymin, xmax, ymax = map(int, box)
            detections.append({
                "class": mapped_class,
                "confidence": float(score),
                "bbox": [xmin, ymin, xmax, ymax],
                "raw_class": raw_class,
            })
        return detections


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_detector(
    backend: str = "auto",
    weights: Optional[Union[str, Path]] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
):
    """Construct a detector.

    Args:
        backend: ``"yolo"`` (fine-tuned RUOD), ``"fasterrcnn"`` (legacy COCO), or
            ``"auto"`` — use the fine-tuned model when weights exist, else fall back
            to the legacy one with a loud warning.
    """
    backend = backend.lower()
    if backend == "fasterrcnn":
        return COCOFasterRCNNDetector(conf_threshold=max(conf_threshold, 0.45),
                                      iou_threshold=iou_threshold)
    if backend == "yolo":
        return UnderwaterYOLODetector(weights=weights, conf_threshold=conf_threshold,
                                      iou_threshold=iou_threshold)

    weights_path = Path(weights) if weights is not None else DEFAULT_YOLO_WEIGHTS
    if weights_path.exists():
        return UnderwaterYOLODetector(weights=weights_path, conf_threshold=conf_threshold,
                                      iou_threshold=iou_threshold)
    logger.warning(
        "No fine-tuned detector at %s; falling back to the legacy COCO Faster R-CNN. "
        "Train the real detector with: python tools/train_detector.py", weights_path
    )
    return COCOFasterRCNNDetector(conf_threshold=max(conf_threshold, 0.45),
                                  iou_threshold=iou_threshold)


# Backwards-compatible alias: anything that imported the old name keeps working, but
# now gets the fine-tuned detector when it is available.
UnderwaterObjectDetector = build_detector


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate_image_with_detections(
    image_input: Union[Path, str, np.ndarray, Image.Image],
    detections: List[Dict[str, Any]],
    output_path: Optional[Union[Path, str]] = None,
) -> Image.Image:
    """Draw bounding boxes and class labels onto an image."""
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        img = Image.fromarray(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    draw = ImageDraw.Draw(img)
    # Scale line width with image size so boxes stay visible on native-resolution frames.
    width = max(2, int(round(min(img.width, img.height) / 400)) + 1)

    for det in detections:
        label = det["class"]
        conf = det["confidence"]
        xmin, ymin, xmax, ymax = det["bbox"]
        color = CLASS_COLORS.get(label, DEFAULT_COLOR)

        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=width)

        text = f"{label.replace('_', ' ').title()}: {conf * 100:.1f}%"
        try:
            font = ImageFont.load_default()
            text_bbox = draw.textbbox((xmin, ymin), text, font=font)
        except AttributeError:
            text_w, text_h = len(text) * 6, 11
            text_bbox = (xmin, ymin, xmin + text_w, ymin + text_h)

        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        banner_ymin = max(0, ymin - text_h - 6)
        draw.rectangle([xmin, banner_ymin, xmin + text_w + 10, banner_ymin + text_h + 6], fill=color)
        draw.text((xmin + 5, banner_ymin + 3), text, fill=(0, 0, 0))

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)

    return img
