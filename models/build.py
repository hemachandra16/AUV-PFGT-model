"""Single source of truth for constructing the PFGT-UIE model.

Before this module existed, ``train.py`` built the model from ``configs/train.yaml``
(``embed_dim=128, num_heads=4``) while ``test.py``, ``validate.py``, ``infer.py``,
``infer_detection.py``, ``smoke_test_amp.py``, ``smoke_test_sm120.py`` and
``profile_bottleneck.py`` all called ``PFGTUIEModel()`` with no arguments and silently
picked up the class default of ``num_heads=1``.

That mismatch was invisible: no weight shape depended on ``num_heads``, so checkpoints
loaded without error — they were simply evaluated with a different attention head count
(and therefore a different softmax temperature) than they were trained with.

Every entry point now builds the model through :func:`build_model`, so train-time and
eval-time architecture can never drift apart again.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from models.model import PFGTUIEModel

logger = logging.getLogger(__name__)

# Resolved relative to the repository root so scripts work from any working directory.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "train.yaml"

# Fallbacks used only if the config file is missing or unreadable. These MUST match the
# values shipped in configs/train.yaml.
FALLBACK_EMBED_DIM = 128
FALLBACK_NUM_HEADS = 4


def _extract(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Pull every architecture knob out of the ``model:`` config section."""
    return {
        "embed_dim": model_cfg.get("embed_dim", FALLBACK_EMBED_DIM),
        "num_heads": model_cfg.get("num_heads", FALLBACK_NUM_HEADS),
        "context_dim": model_cfg.get("context_dim", 64),
        "low_mlp_ratio": model_cfg.get("low_mlp_ratio", 4),
        "high_mlp_ratio": model_cfg.get("high_mlp_ratio", 2),
        "use_global_correction": model_cfg.get("use_global_correction", True),
        "use_physics_priors": model_cfg.get("use_physics_priors", True),
    }


def load_model_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Read the ``model:`` section of the training config.

    Falls back to the documented defaults (and logs a warning) if the file or PyYAML is
    unavailable, so inference scripts never hard-fail on a missing config.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH

    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML unavailable; using fallback model config.")
        return _extract({})

    if not path.exists():
        logger.warning("Model config not found at %s; using fallback model config.", path)
        return _extract({})

    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    model_cfg = cfg.get("model", {}) or {}
    return _extract(model_cfg)


def build_model(
    config: Optional[Union[str, Path, Dict[str, Any]]] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> PFGTUIEModel:
    """Construct a :class:`PFGTUIEModel` with the project's configured architecture.

    Args:
        config: Path to a YAML config, an already-loaded full config dict, or ``None``
            to use ``configs/train.yaml``.
        device: Optional device to move the model to.

    Returns:
        The constructed model.
    """
    if isinstance(config, dict):
        # An already-loaded full training config was passed in (train.py does this).
        model_cfg = config.get("model", {}) or {}
        params = _extract(model_cfg)
    else:
        params = load_model_config(config)

    model = PFGTUIEModel(**params)
    logger.info(
        "Built PFGTUIEModel(embed_dim=%d, num_heads=%d, context_dim=%d, low_mlp=%d, "
        "high_mlp=%d, global_correction=%s, physics_priors=%s)",
        params["embed_dim"], params["num_heads"], params["context_dim"],
        params["low_mlp_ratio"], params["high_mlp_ratio"],
        params["use_global_correction"], params["use_physics_priors"],
    )

    if device is not None:
        model = model.to(device)
    return model
