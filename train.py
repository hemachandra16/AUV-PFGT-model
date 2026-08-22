"""PFGT-UIE Training Script.

Trains the Physics-aware Frequency-Guided Transformer for Underwater Image
Enhancement on the UIEB dataset with:
  - Automatic Mixed Precision (AMP)
  - Gradient clipping
  - Warmup + cosine annealing LR schedule
  - Best + latest checkpoint saving
  - Resume from checkpoint
  - Early stopping
  - TensorBoard logging (loss, LR, grad norm, GPU mem, PSNR, SSIM)
  - Reproducibility via seed_everything()

Usage:
    python train.py --config configs/train.yaml
    python train.py --config configs/train.yaml --resume checkpoints/latest.pt
    python train.py --config configs/train.yaml --epochs 5 --debug-mode
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# Try loading PyYAML for config support
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

from data.dataset import UIEBDataset, create_dataloader, get_splits
from metrics import compute_psnr, compute_ssim
from models.loss import PFGTLoss
from models.build import build_model
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.logging_utils import setup_logger
from utils.seed import seed_everything


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _deep_update(base: dict, override: dict) -> dict:
    """Recursively update base dict with override values."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(config_path: Optional[str]) -> dict:
    """Load YAML config; fall back to empty dict if YAML unavailable."""
    if config_path is None or not HAS_YAML:
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the PFGT-UIE model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="configs/train.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override total training epochs from config")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size from config")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate from config")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume an INTERRUPTED run from. Restores "
                             "model, optimizer, scaler, epoch and global_step, so the LR "
                             "schedule continues where it left off.")
    parser.add_argument("--init-from", type=str, default=None,
                        help="Initialise MODEL WEIGHTS ONLY from a checkpoint and start a fresh "
                             "run (epoch 0, step 0, new optimizer, new LR schedule). This is "
                             "what fine-tuning needs: --resume would inherit the previous run's "
                             "global_step, and since the LR is computed as global_step/"
                             "total_steps, a short fine-tune schedule would be driven past its "
                             "end and swing the LR back up toward base_lr.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Override checkpoint directory from config")
    parser.add_argument("--debug-mode", action="store_true",
                        help="Run a single batch per epoch for debugging")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Training device")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable Automatic Mixed Precision")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override random seed from config")
    parser.add_argument("--num-workers", type=int, default=None,
                        help="DataLoader worker processes")
    parser.add_argument("--val-every", type=int, default=1,
                        help="Run validation every N epochs (use >1 to speed up CPU training)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def get_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# LR warmup + cosine annealing
# ---------------------------------------------------------------------------

def get_warmup_cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    eta_min: float,
) -> float:
    """Compute learning rate for linear warmup then cosine annealing."""
    if step < warmup_steps:
        return base_lr * max(step, 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    cosine_lr = eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * progress))
    return cosine_lr


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def build_dataloaders(
    cfg: dict,
    batch_size: int,
    num_workers: int,
    debug_mode: bool,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation DataLoaders from the UIEB dataset."""
    dl_cfg = cfg.get("dataloader", {})

    # The 90/10 split now lives in data/dataset.py::get_splits so that validate.py and
    # test.py --mode dataset score the SAME held-out 10% and never leak training data.
    train_dataset, val_dataset = get_splits(cfg)

    logger.info(
        "Dataset split: %d train / %d validation images (shared get_splits, seed=42)",
        len(train_dataset), len(val_dataset),
    )

    pin_memory = dl_cfg.get("pin_memory", True) and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=not debug_mode,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Validation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_validation(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: PFGTLoss,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    """Run one full validation pass and return aggregated metrics."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_batches = 0

    for inputs, targets in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(inputs)
            losses = criterion(outputs, targets)

        total_loss += float(losses["total_loss"].item())
        total_psnr += compute_psnr(outputs, targets)
        total_ssim += compute_ssim(outputs, targets)
        n_batches += 1

    model.train()
    n = max(n_batches, 1)
    return {
        "val_loss": total_loss / n,
        "psnr": total_psnr / n,
        "ssim": total_ssim / n,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load YAML config then override with CLI args
    cfg = load_config(args.config)

    train_cfg = cfg.get("training", {})
    sched_cfg = cfg.get("scheduler", {})
    opt_cfg = cfg.get("optimizer", {})
    ckpt_cfg = cfg.get("checkpoint", {})
    loss_cfg = cfg.get("loss", {})
    dl_cfg = cfg.get("dataloader", {})
    log_cfg = cfg.get("logging", {})

    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    seed_everything(seed)

    # ---- Setup logger ----
    log_dir = log_cfg.get("log_dir", "logs")
    setup_logger(log_dir=log_dir, log_file="train.log")
    logger.info("PFGT-UIE Training — seed=%d", seed)

    # ---- Device ----
    device = get_device(args.device)
    logger.info("Training on device: %s", device)

    # ---- Hyperparameters ----
    total_epochs = args.epochs if args.epochs is not None else train_cfg.get("epochs", 150)
    batch_size = args.batch_size if args.batch_size is not None else dl_cfg.get("batch_size", 4)
    base_lr = args.lr if args.lr is not None else opt_cfg.get("lr", 1e-4)
    use_amp = (not args.no_amp) and train_cfg.get("amp", True) and (device.type == "cuda")
    grad_clip_norm = train_cfg.get("grad_clip_norm", 1.0)
    log_every = train_cfg.get("log_every_steps", 10)
    num_workers = args.num_workers if args.num_workers is not None else dl_cfg.get("num_workers", 4)
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else ckpt_cfg.get("dir", "checkpoints")
    warmup_epochs = sched_cfg.get("warmup_epochs", 5)
    eta_min = sched_cfg.get("eta_min", 1e-6)
    val_every = args.val_every  # validate every N epochs

    # Early stopping settings
    es_cfg = cfg.get("early_stopping", {})
    es_enabled = es_cfg.get("enabled", True)
    es_patience = es_cfg.get("patience", 20)
    es_min_delta = es_cfg.get("min_delta", 0.01)
    # early_stopping.metric is now honoured rather than hardcoded to PSNR.
    es_metric = str(es_cfg.get("metric", "psnr")).lower()
    if es_metric not in ("psnr", "ssim"):
        logger.warning("Unsupported early_stopping.metric %r; falling back to 'psnr'.", es_metric)
        es_metric = "psnr"

    save_every_epochs = train_cfg.get("save_every_epochs", 0) or 0

    logger.info(
        "Config: epochs=%d  bs=%d  lr=%.2e  amp=%s  workers=%d  grad_clip=%.1f",
        total_epochs, batch_size, base_lr, use_amp, num_workers, grad_clip_norm,
    )

    # ---- TensorBoard ----
    tb_dir = log_cfg.get("tensorboard_dir", "logs/tensorboard")
    writer = SummaryWriter(log_dir=tb_dir)
    logger.info("TensorBoard logs -> %s", tb_dir)

    # ---- Model ----
    # Built through the shared helper so train-time and eval-time architecture
    # can never drift apart (see models/build.py).
    model = build_model(cfg, device=device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s", f"{n_params:,}")

    # ---- Loss ----
    criterion = PFGTLoss(
        lambda_l1=loss_cfg.get("lambda_l1", 1.0),
        lambda_ssim=loss_cfg.get("lambda_ssim", 0.5),
        lambda_perceptual=loss_cfg.get("lambda_perceptual", 0.1),
        lambda_frequency=loss_cfg.get("lambda_frequency", 0.1),
        lambda_dc=loss_cfg.get("lambda_dc", 0.0),
    ).to(device)

    # ---- Optimizer ----
    weight_decay = opt_cfg.get("weight_decay", 1e-4)
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=betas,
    )

    # ---- AMP Scaler ----
    scaler = GradScaler(device=device.type, enabled=use_amp)

    # ---- Data ----
    train_loader, val_loader = build_dataloaders(cfg, batch_size, num_workers, args.debug_mode)

    # ---- LR Scheduler ----
    # We will manually handle warmup; cosine annealing starts after warmup.
    steps_per_epoch = len(train_loader)
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    global_step = 0

    # ---- Resume ----
    start_epoch = 0
    best_metric = 0.0
    es_counter = 0

    if args.init_from is not None and args.resume is not None:
        raise SystemExit("--init-from and --resume are mutually exclusive: the first starts a "
                         "fresh schedule, the second continues an existing one.")

    if args.init_from is not None:
        logger.info("Initialising weights from: %s (fresh optimizer and LR schedule)",
                    args.init_from)
        load_checkpoint(args.init_from, model=model, device=device)
        logger.info("Fine-tune mode: starting at epoch 0, step 0, lr=%.2e", base_lr)

    if args.resume is not None:
        logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
        )
        start_epoch = ckpt.get("epoch", 0)
        global_step = ckpt.get("step", 0)
        best_metric = ckpt.get("metrics", {}).get(es_metric, 0.0)
        es_counter = ckpt.get("es_counter", 0)
        logger.info("Resumed from epoch %d (global_step=%d, best_%s=%.4f, es_counter=%d)",
                    start_epoch, global_step, es_metric, best_metric, es_counter)

    os.makedirs(checkpoint_dir, exist_ok=True)

    # ---- Training Loop ----
    model.train()
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        running_loss = 0.0
        running_l1 = 0.0
        running_ssim_loss = 0.0
        running_perc = 0.0
        running_freq = 0.0
        running_dc = 0.0
        total_steps_epoch = 0

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # --- LR warmup / cosine schedule (per-step) ---
            current_lr = get_warmup_cosine_lr(
                global_step, warmup_steps, total_steps, base_lr, eta_min
            )
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=use_amp):
                outputs = model(inputs)
                losses = criterion(outputs, targets)

            total_loss = losses["total_loss"]

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(total_loss.item())
            running_l1 += float(losses["l1_loss"].item())
            running_ssim_loss += float(losses["ssim_loss"].item())
            running_perc += float(losses["perceptual_loss"].item())
            running_freq += float(losses["frequency_loss"].item())
            running_dc += float(losses["dc_loss"].item())
            total_steps_epoch += 1
            global_step += 1

            # --- Logging ---
            if global_step % log_every == 0:
                logger.info(
                    "epoch=%d/%d  step=%d  loss=%.6f  l1=%.6f  ssim=%.6f  "
                    "perc=%.6f  freq=%.6f  dc=%.6f  grad_norm=%.4f  lr=%.2e",
                    epoch + 1, total_epochs, step + 1,
                    float(total_loss.item()),
                    float(losses["l1_loss"].item()),
                    float(losses["ssim_loss"].item()),
                    float(losses["perceptual_loss"].item()),
                    float(losses["frequency_loss"].item()),
                    float(losses["dc_loss"].item()),
                    float(grad_norm),
                    current_lr,
                )
                writer.add_scalar("train/loss", float(total_loss.item()), global_step)
                writer.add_scalar("train/l1_loss", float(losses["l1_loss"].item()), global_step)
                writer.add_scalar("train/ssim_loss", float(losses["ssim_loss"].item()), global_step)
                writer.add_scalar("train/perceptual_loss", float(losses["perceptual_loss"].item()), global_step)
                writer.add_scalar("train/frequency_loss", float(losses["frequency_loss"].item()), global_step)
                writer.add_scalar("train/dc_loss", float(losses["dc_loss"].item()), global_step)
                writer.add_scalar("train/grad_norm", float(grad_norm), global_step)
                writer.add_scalar("train/lr", current_lr, global_step)

                if device.type == "cuda":
                    gpu_mem_mb = torch.cuda.memory_allocated(device) / 1e6
                    writer.add_scalar("train/gpu_mem_mb", gpu_mem_mb, global_step)

            if args.debug_mode:
                break  # One batch only in debug mode

        # ---- Epoch summary ----
        epoch_duration = time.time() - epoch_start
        avg_loss = running_loss / max(total_steps_epoch, 1)
        logger.info(
            "=== Epoch %d/%d finished | avg_loss=%.6f | time=%.1fs ===",
            epoch + 1, total_epochs, avg_loss, epoch_duration,
        )
        writer.add_scalar("epoch/train_loss", avg_loss, epoch + 1)
        writer.add_scalar("epoch/duration_s", epoch_duration, epoch + 1)

        # ---- Validation (every val_every epochs) ----
        do_validate = ((epoch + 1) % val_every == 0) or (epoch + 1 == total_epochs)
        # training.save_every_epochs now actually produces checkpoints/epoch_N.pt
        # snapshots — extra recovery points for long unattended runs.
        save_periodic = bool(save_every_epochs) and ((epoch + 1) % save_every_epochs == 0)
        if do_validate:
            val_metrics = run_validation(model, val_loader, criterion, device, use_amp)
            psnr = val_metrics["psnr"]
            ssim = val_metrics["ssim"]
            val_loss = val_metrics["val_loss"]

            logger.info(
                "  Validation -> val_loss=%.6f  PSNR=%.4f dB  SSIM=%.4f",
                val_loss, psnr, ssim,
            )
            writer.add_scalar("val/loss", val_loss, epoch + 1)
            writer.add_scalar("val/psnr", psnr, epoch + 1)
            writer.add_scalar("val/ssim", ssim, epoch + 1)

            # ---- Checkpoint ----
            # Track whichever metric early_stopping.metric names (higher is better).
            current_metric = val_metrics[es_metric]
            is_best = current_metric > best_metric + es_min_delta
            if is_best:
                best_metric = current_metric
                es_counter = 0
            else:
                es_counter += 1

            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=None,
                epoch=epoch + 1,
                step=global_step,
                metrics={**val_metrics, "avg_train_loss": avg_loss},
                is_best=is_best,
                es_counter=es_counter,
                save_periodic=save_periodic,
            )
        else:
            # Save latest checkpoint even when skipping validation
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=None,
                epoch=epoch + 1,
                step=global_step,
                metrics={"avg_train_loss": avg_loss},
                is_best=False,
                es_counter=es_counter,
                save_periodic=save_periodic,
            )

        # ---- Early stopping ----
        if do_validate and es_enabled and es_counter >= es_patience:
            logger.info(
                "Early stopping triggered: %s did not improve for %d validation rounds.",
                es_metric.upper(), es_patience
            )
            break

    writer.close()
    logger.info("Training complete. Best %s: %.4f", es_metric.upper(), best_metric)
    logger.info("Best checkpoint: %s/best.pt", checkpoint_dir)


if __name__ == "__main__":
    main()
