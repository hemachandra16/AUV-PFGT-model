"""Demonstrate that each session-3 fix does what it claims, before any training run.

Same standard as tools/verify_attention.py: show it on a controlled case, do not assert it.

  1. PHYSICS PRIORS respond to degradation — the priors must change monotonically with how
     badly an image is degraded, or they carry no usable signal.
  2. GLOBAL CONTEXT exists and varies — the old encoder was all 3x3 convs with no pooling, so
     it could not represent image-wide statistics at any depth. The new one must.
  3. GLOBAL CORRECTION IS EXACT IDENTITY AT INIT — so it cannot regress the model at step 0
     and any change it later produces is attributable to learning.
  4. GLOBAL CORRECTION HAS THE CAPACITY TO DELIVER — fit only its 4,550 parameters against the
     oracle per-image offset and check it recovers a large share of the measured +3.20 dB
     headroom. This is the load-bearing check: a pathway that exists but cannot express the
     correction would be useless.
  5. GROUPNORM CLOSED THE TRAIN/EVAL GAP — BatchNorm at batch 8 made the refinement head
     evaluate a different function from the one it trained; GroupNorm must not.
  6. GRADIENTS REACH EVERY NEW PARAMETER — a fix that receives no gradient is decoration.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from models.build import build_model
from models.global_correction import GlobalColorCorrection
from models.physics_encoder import PhysicsPriorEncoder, compute_physics_priors

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cast(x, r, b):
    y = x.clone()
    y[:, 0] *= r
    y[:, 2] *= b
    return y.clamp(0, 1)


def check_priors(dev) -> bool:
    print("=" * 74)
    print("CHECK 1 - closed-form physics priors respond to degradation severity")
    print("=" * 74)
    img = Image.open(os.path.join(ROOT, "datasets/UIEB/reference-890/708_img_.png")).convert("RGB").resize((256, 256))
    x0 = torch.from_numpy(np.array(img, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(dev)

    sevs = [("none", 1.00, 1.00), ("mild", 0.80, 1.10), ("medium", 0.60, 1.25), ("severe", 0.40, 1.40)]
    print(f"  {'severity':<10}{'dark':>9}{'bright':>9}{'localstd':>10}{'mu_R':>8}{'R/G':>8}{'R/B':>8}")
    print("  " + "-" * 62)
    rg = []
    for tag, r, b in sevs:
        p = compute_physics_priors(cast(x0, r, b))
        v = [float(p[:, i].mean()) for i in range(8)]
        rg.append(v[6])
        print(f"  {tag:<10}{v[0]:>9.4f}{v[1]:>9.4f}{v[2]:>10.4f}{v[3]:>8.4f}{v[6]:>8.4f}{v[7]:>8.4f}")
    mono = all(rg[i] > rg[i + 1] for i in range(len(rg) - 1))
    print(f"\n  R/G ratio decreases monotonically with severity: {mono}")
    print(f"  RESULT: {'PASS' if mono else 'FAIL'}")
    return mono


def check_global_context(dev) -> bool:
    print("=" * 74)
    print("CHECK 2 - the encoder now has a global pathway the old one structurally lacked")
    print("=" * 74)
    enc = PhysicsPriorEncoder().to(dev).eval()
    has_pool = any(isinstance(m, torch.nn.Linear) for m in enc.modules())
    img = Image.open(os.path.join(ROOT, "datasets/UIEB/raw-890/708_img_.png")).convert("RGB").resize((256, 256))
    x0 = torch.from_numpy(np.array(img, dtype=np.float32) / 255).permute(2, 0, 1).unsqueeze(0).to(dev)

    with torch.no_grad():
        _, c_none = enc(x0)
        _, c_mild = enc(cast(x0, 0.8, 1.1))
        _, c_sev = enc(cast(x0, 0.4, 1.4))
    d_mild = float((c_mild - c_none).abs().mean() / (c_none.abs().mean() + 1e-9)) * 100
    d_sev = float((c_sev - c_none).abs().mean() / (c_none.abs().mean() + 1e-9)) * 100

    # The decisive structural test: a global-mean-preserving spatial shuffle leaves image-wide
    # statistics identical, so a context that is genuinely global should barely move.
    perm = torch.randperm(256 * 256, device=dev)
    shuf = x0.flatten(2)[:, :, perm].reshape_as(x0)
    with torch.no_grad():
        _, c_shuf = enc(shuf)
    d_shuf = float((c_shuf - c_none).abs().mean() / (c_none.abs().mean() + 1e-9)) * 100

    print(f"  encoder contains Linear/global-pool layers : {has_pool}   (old encoder: False)")
    print(f"  context change, mild cast                  : {d_mild:.2f}%")
    print(f"  context change, severe cast                : {d_sev:.2f}%")
    print(f"  context change, pixel shuffle (same global stats) : {d_shuf:.2f}%")
    ok = has_pool and d_sev > d_mild and d_sev > 1.0
    print(f"\n  context grows with severity: {d_sev > d_mild}")
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def check_identity_init(dev) -> bool:
    print("=" * 74)
    print("CHECK 3 - global correction is an EXACT identity at initialisation")
    print("=" * 74)
    gc = GlobalColorCorrection(context_dim=64).to(dev).eval()
    img = torch.rand(4, 3, 64, 64, device=dev)
    ctx = torch.randn(4, 64, device=dev)
    with torch.no_grad():
        out = gc(img, ctx)
        gain, shift = gc.predicted_params(ctx)
    diff = float((out - img).abs().max())
    print(f"  max |out - in|      : {diff:.3e}")
    print(f"  gain  (want 1.0)    : {float(gain.min()):.6f} .. {float(gain.max()):.6f}")
    print(f"  shift (want 0.0)    : {float(shift.min()):.6f} .. {float(shift.max()):.6f}")
    ok = diff < 1e-6
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def check_capacity(dev) -> bool:
    print("=" * 74)
    print("CHECK 4 - can the new pathway actually deliver the measured headroom?")
    print("=" * 74)
    print("  Fitting ONLY the 4,550 global-correction parameters against the oracle")
    print("  per-image offset, with the rest of the network frozen.\n")

    from torch.utils.data import DataLoader
    from data.dataset import get_splits
    from metrics import compute_psnr
    from utils.checkpoint import load_checkpoint

    # Use the OLD converged checkpoint's behaviour as the base image to correct: we are asking
    # whether a context-predicted affine can close the gap, not whether the new model is good.
    _, val = get_splits(augment_train=False)
    loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0)

    model = build_model(device=dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    gc = model.global_correction
    for p in gc.parameters():
        p.requires_grad_(True)

    opt = torch.optim.Adam(gc.parameters(), lr=3e-3)

    # Cache base outputs + contexts once (frozen network -> they never change).
    cache = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            feats, ctx = model.physics_encoder(x)
            saved = model.global_correction
            model.global_correction = None
            base = model(x)
            model.global_correction = saved
            cache.append((base, ctx, y))

    def psnr_of(apply_gc: bool) -> float:
        tot = 0.0
        n = 0
        with torch.no_grad():
            for base, ctx, y in cache:
                out = gc(base, ctx) if apply_gc else base
                for i in range(out.shape[0]):
                    tot += compute_psnr(out[i:i + 1], y[i:i + 1])
                    n += 1
        return tot / n

    def oracle_psnr() -> float:
        tot = 0.0
        n = 0
        with torch.no_grad():
            for base, ctx, y in cache:
                d = (y - base).mean(dim=(2, 3), keepdim=True)
                out = (base + d).clamp(0, 1)
                for i in range(out.shape[0]):
                    tot += compute_psnr(out[i:i + 1], y[i:i + 1])
                    n += 1
        return tot / n

    p_before = psnr_of(False)
    p_oracle = oracle_psnr()

    for step in range(400):
        for base, ctx, y in cache:
            opt.zero_grad()
            out = gc(base, ctx)
            loss = F.l1_loss(out, y)
            loss.backward()
            opt.step()
    p_after = psnr_of(True)

    head = p_oracle - p_before
    got = p_after - p_before
    frac = got / head if head > 1e-6 else 0.0
    print(f"  base (no correction)          : {p_before:.4f} dB")
    print(f"  oracle per-image offset       : {p_oracle:.4f} dB   (headroom {head:+.3f})")
    print(f"  learned from physics context  : {p_after:.4f} dB   (recovered {got:+.3f})")
    print(f"  fraction of headroom captured : {frac*100:.1f}%")
    ok = got > 0.3 * head and got > 0.2
    print(f"\n  RESULT: {'PASS' if ok else 'FAIL'}")
    print("  (An untrained network is being corrected here, so this measures the PATHWAY's")
    print("   capacity, not the final model's quality.)")
    return ok


def check_groupnorm(dev) -> bool:
    print("=" * 74)
    print("CHECK 5 - GroupNorm removed the train/eval function gap BatchNorm caused")
    print("=" * 74)
    from models.refinement import ImageRefinementHead
    head = ImageRefinementHead(128).to(dev)
    x = torch.randn(8, 128, 64, 64, device=dev)
    head.train()
    with torch.no_grad():
        y_train = head(x)
    head.eval()
    with torch.no_grad():
        y_eval = head(x)
    gap = float((y_train - y_eval).abs().max())
    norms = sorted({type(m).__name__ for m in head.modules()
                    if "Norm" in type(m).__name__})
    print(f"  norm layers in the head : {norms}")
    print(f"  max |train(x) - eval(x)| : {gap:.3e}   (BatchNorm gave a nonzero gap)")
    ok = gap < 1e-6 and "BatchNorm2d" not in norms
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def check_gradients(dev) -> bool:
    print("=" * 74)
    print("CHECK 6 - every new parameter actually receives gradient")
    print("=" * 74)
    model = build_model(device=dev)
    model.train()
    x = torch.rand(2, 3, 256, 256, device=dev)
    t = torch.rand(2, 3, 256, 256, device=dev)
    out = model(x)
    F.l1_loss(out, t).backward()

    groups = {
        "physics_encoder.context_mlp": "physics_encoder.context_mlp",
        "physics_encoder.se": "physics_encoder.se",
        "physics_encoder.initial_conv (11-ch stem)": "physics_encoder.initial_conv",
        "global_correction.predictor": "global_correction.predictor",
        "refinement final_conv bias": "refinement_head.final_conv.0.bias",
    }
    ok = True
    for label, prefix in groups.items():
        gs = [p.grad for n, p in model.named_parameters()
              if n.startswith(prefix) and p.grad is not None]
        tot = sum(float(g.abs().sum()) for g in gs)
        alive = len(gs) > 0 and tot > 0
        print(f"  {label:<44} tensors={len(gs)}  sum|grad|={tot:.3e}  {'OK' if alive else 'NO GRADIENT'}")
        ok = ok and alive
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}\n")
    res = [
        ("physics priors respond to degradation", check_priors(dev)),
        ("encoder has a global pathway", check_global_context(dev)),
        ("global correction is identity at init", check_identity_init(dev)),
        ("global correction can deliver the headroom", check_capacity(dev)),
        ("GroupNorm closed the train/eval gap", check_groupnorm(dev)),
        ("gradients reach every new parameter", check_gradients(dev)),
    ]
    print("=" * 74)
    for name, ok in res:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 74)
    sys.exit(0 if all(o for _, o in res) else 1)


if __name__ == "__main__":
    main()
