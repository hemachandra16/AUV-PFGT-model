"""Checkpoint save/load utilities for PFGT-UIE.

Implements two checkpoint strategies:
- latest.pt: overwritten every epoch — allows resuming after any crash
- best.pt: only saved when validation metric improves — preserves best model
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    scheduler: Optional[Any],
    epoch: int,
    step: int,
    metrics: Dict[str, float],
    is_best: bool = False,
    es_counter: int = 0,
    save_periodic: bool = False,
) -> None:
    """Save model state as latest and optionally as best / periodic checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "es_counter": es_counter,
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()

    latest_path = checkpoint_dir / "latest.pt"
    torch.save(state, latest_path)
    logger.info("Saved latest checkpoint to %s (epoch=%d)", latest_path, epoch)

    if is_best:
        best_path = checkpoint_dir / "best.pt"
        torch.save(state, best_path)
        logger.info("Saved best checkpoint to %s (epoch=%d)", best_path, epoch)

    if save_periodic:
        periodic_path = checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(state, periodic_path)
        logger.info("Saved periodic checkpoint to %s", periodic_path)


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[Any] = None,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load a checkpoint and restore model (and optionally optimizer/scaler/scheduler) state.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model:           Model to load state into.
        optimizer:       If provided, optimizer state is restored.
        scaler:          If provided, AMP GradScaler state is restored.
        scheduler:       If provided, LR scheduler state is restored.
        device:          Device to map tensors to.

    Returns:
        The full checkpoint dictionary (contains epoch, step, metrics, etc.)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info("Loaded model weights from %s (epoch=%d)", checkpoint_path, checkpoint.get("epoch", -1))

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Restored optimizer state.")

    if scaler is not None and "scaler_state_dict" in checkpoint:
        if checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.info("Restored AMP scaler state.")
        else:
            logger.warning("AMP scaler state in checkpoint is empty (likely saved from a CPU/no-AMP run). Skipping scaler restore.")

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Restored LR scheduler state.")

    return checkpoint
