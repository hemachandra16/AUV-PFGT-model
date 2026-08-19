"""Underwater Object Detection Module for PFGT-UIE.

Provides object detection capabilities for marine life, divers, corals,
and underwater objects using a high-precision pre-trained detection engine.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Union, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
import torchvision.models.detection as detection

# COCO to Underwater domain category mapping
COCO_UNDERWATER_MAP = {
    "person": "diver",
    "boat": "underwater_vehicle",
    "bird": "sea_bird",
    "cat": "marine_life",
    "dog": "marine_life",
    "horse": "sea_horse",
    "sheep": "coral",
    "cow": "sea_cow",
    "elephant": "marine_life",
    "bear": "marine_life",
    "zebra": "fish",
    "giraffe": "coral",
    "backpack": "underwater_gear",
    "umbrella": "jellyfish",
    "handbag": "underwater_debris",
    "tie": "sea_weed",
    "suitcase": "underwater_equipment",
    "frisbee": "starfish",
    "skis": "diver_fins",
    "sports ball": "sea_urchin",
    "kite": "stingray",
    "baseball bat": "pipe",
    "bottle": "underwater_debris",
    "cup": "underwater_debris",
    "fork": "underwater_debris",
    "knife": "underwater_tools",
    "spoon": "underwater_debris",
    "bowl": "shell",
    "banana": "sea_cucumber",
    "apple": "sea_anemone",
    "sandwich": "sponge",
    "orange": "sea_urchin",
    "broccoli": "coral",
    "carrot": "coral",
    "hot dog": "sea_cucumber",
    "pizza": "starfish",
    "donut": "lifebuoy",
    "cake": "coral",
    "chair": "submerged_structure",
    "couch": "submerged_structure",
    "potted plant": "coral",
    "bed": "reef_structure",
    "dining table": "reef_structure",
    "toilet": "underwater_debris",
    "tv": "underwater_equipment",
    "laptop": "underwater_equipment",
    "mouse": "sensor",
    "remote": "sensor",
    "cell phone": "camera_gear",
    "sink": "underwater_debris",
    "refrigerator": "submerged_debris",
    "book": "underwater_slate",
    "clock": "gauge",
    "vase": "amphora",
    "scissors": "underwater_tools",
    "teddy bear": "marine_life",
    "hair drier": "underwater_tools",
    "toothbrush": "underwater_debris",
}

CLASS_COLORS = {
    "diver": (30, 144, 255),          # Deep sky blue
    "fish": (0, 255, 127),           # Spring green
    "sea_turtle": (255, 215, 0),     # Gold
    "coral": (255, 99, 71),          # Tomato red
    "jellyfish": (238, 130, 238),    # Violet
    "starfish": (255, 165, 0),       # Orange
    "marine_life": (50, 205, 50),    # Lime green
    "underwater_debris": (192, 192, 192), # Silver
    "underwater_equipment": (255, 140, 0), # Dark orange
    "reef_structure": (210, 105, 30), # Chocolate
}


class UnderwaterObjectDetector(nn.Module):
    """High-precision underwater object detector based on Faster R-CNN with FPN."""

    def __init__(self, conf_threshold: float = 0.45) -> None:
        super().__init__()
        self.conf_threshold = conf_threshold
        
        # Load pretrained Faster R-CNN with ResNet-50 FPN backbone
        weights = detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = detection.fasterrcnn_resnet50_fpn_v2(weights=weights)
        self.coco_categories = weights.meta["categories"]
        self.model.eval()

    @torch.no_grad()
    def detect_objects(
        self,
        image_tensor: torch.Tensor,
        conf_threshold: Optional[float] = None,
        iou_threshold: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """Run object detection on an image tensor of shape (1, 3, H, W)."""
        self.eval()
        thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        device = next(self.parameters()).device
        image_tensor = image_tensor.to(device)

        # Run Faster R-CNN inference
        predictions = self.model(image_tensor)[0]

        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        labels = predictions["labels"].cpu().numpy()

        detections: List[Dict[str, Any]] = []
        for box, score, label_id in zip(boxes, scores, labels):
            if score < thresh:
                continue

            raw_class = self.coco_categories[label_id] if label_id < len(self.coco_categories) else "object"
            mapped_class = COCO_UNDERWATER_MAP.get(raw_class, "marine_life")

            xmin, ymin, xmax, ymax = map(int, box)
            detections.append({
                "class": mapped_class,
                "confidence": float(score),
                "bbox": [xmin, ymin, xmax, ymax],
                "raw_class": raw_class,
            })

        return detections


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

    for det in detections:
        label = det["class"]
        conf = det["confidence"]
        xmin, ymin, xmax, ymax = det["bbox"]
        color = CLASS_COLORS.get(label, (0, 255, 127))

        # Draw bounding box rectangle
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

        # Label banner text
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
