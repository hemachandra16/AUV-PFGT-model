"""Evidence that the physics-guided attention fix is real, not cosmetic.

Runs four checks:

  1. CONVEX HULL — the mathematical heart of the bug. With Q = K = V (the old code),
     softmax(QK'/sqrt(d))V is a row-stochastic matrix applied to the tokens, so every
     output value is a weighted average of input values in the same channel and can
     never leave [min, max] of that channel. The fixed module has learned V and output
     projections, so it can.

  2. PHYSICS SENSITIVITY — changing only the physics feature map must change the output.
     If it does not, the physics guidance is decorative.

  3. HEAD COUNT MATTERS — num_heads must actually change the computation (the old module
     ignored it beyond a scale divisor, which is why the silent train/eval mismatch went
     unnoticed).

  4. FULL MODEL — forward + backward at the real training shape, no NaNs, plus peak VRAM.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from models.attention.physics_attention import PhysicsGuidedAttention
from models.build import build_model


class OldStyleAttention(torch.nn.Module):
    """Faithful reproduction of the PREVIOUS physics_attention.py.

    Note what it does and does not own: there are no Q/K/V/output projections at all,
    so its entire learnable parameter set is the 1-channel physics projection plus the
    physics_scale scalar. The token pathway is parameter-free.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        self.physics_projection = torch.nn.Conv2d(64, 1, kernel_size=1, bias=True)
        self.physics_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, q, k, v, physics):
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        projected = self.physics_projection(physics)
        n = q.shape[1]
        g = int(n ** 0.5)
        pooled = F.adaptive_avg_pool2d(projected, (g, g))
        t = pooled.flatten(2).squeeze(1)
        t = (t - t.mean(1, keepdim=True)) / (t.std(1, keepdim=True) + 1e-6)
        t = t.unsqueeze(-1)
        bias = torch.bmm(t, t.transpose(1, 2))
        return torch.matmul(F.softmax(scores + self.physics_scale * bias, dim=-1), v)


def _fit(module, tokens, physics, target, steps=400, lr=5e-2):
    """Fit a module to a target and return the final MSE."""
    params = [p for p in module.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr) if params else None
    last = None
    for _ in range(steps):
        out = module(tokens, tokens, tokens, physics)
        loss = F.mse_loss(out, target)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        last = float(loss)
    return last


def check_convex_hull(device: str) -> bool:
    print("=" * 72)
    print("CHECK 1 - convex-hull limitation (the actual bug)")
    print("=" * 72)

    torch.manual_seed(0)
    B, N, C = 2, 64, 128
    tokens = torch.randn(B, N, C, device=device)
    physics = torch.randn(B, 64, 32, 32, device=device)

    # Target = a uniform additive shift, i.e. exactly the "global colour cast removal"
    # operation underwater enhancement needs. It lies wholly OUTSIDE the convex hull of
    # the input tokens, so a convex-blending operator provably cannot reach it.
    target = tokens + 3.0

    torch.manual_seed(7)
    old = OldStyleAttention(C, num_heads=4).to(device)
    torch.manual_seed(7)
    new = PhysicsGuidedAttention(embed_dim=C, num_heads=4).to(device)

    n_old = sum(p.numel() for p in old.parameters())
    n_new = sum(p.numel() for p in new.parameters())

    old_loss = _fit(old, tokens, physics, target)
    new_loss = _fit(new, tokens, physics, target)

    # Where the outputs actually land relative to the input range.
    with torch.no_grad():
        o_out = old(tokens, tokens, tokens, physics)
        n_out = new(tokens, tokens, tokens, physics)

    print(f"  task: learn tokens -> tokens + 3.0 (a pure colour shift, outside the hull)")
    print(f"  OLD module learnable params : {n_old:5d}  (no Q/K/V projections at all)")
    print(f"  NEW module learnable params : {n_new:5d}")
    print(f"  input  mean : {tokens.mean():+.4f}   target mean : {target.mean():+.4f}")
    print(f"  OLD output mean after fitting: {o_out.mean():+.4f}   final MSE: {old_loss:9.5f}")
    print(f"  NEW output mean after fitting: {n_out.mean():+.4f}   final MSE: {new_loss:9.5f}")
    print(f"  -> the old operator stays pinned near the input mean: it can only average")
    print(f"     existing tokens, so a global shift is unreachable. This is why enhanced")
    print(f"     output looked like a near-copy of the hazy input.")

    ok = old_loss > 5.0 and new_loss < old_loss / 5.0
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def check_physics_sensitivity(device: str) -> bool:
    print("=" * 72)
    print("CHECK 2 — output responds to the physics feature map")
    print("=" * 72)

    torch.manual_seed(1)
    B, N, C = 2, 64, 128
    tokens = torch.randn(B, N, C, device=device)
    attn = PhysicsGuidedAttention(embed_dim=C, num_heads=4).to(device)
    # Push physics_scale away from init so the bias is clearly active.
    with torch.no_grad():
        attn.physics_scale.fill_(4.0)
        attn.physics_projection.weight.mul_(20.0)

    physics_a = torch.randn(B, 64, 32, 32, device=device)
    physics_b = torch.randn(B, 64, 32, 32, device=device)

    with torch.no_grad():
        out_a = attn(tokens, tokens, tokens, physics_a)
        out_b = attn(tokens, tokens, tokens, physics_b)

    delta = (out_a - out_b).abs().mean().item()
    scale = out_a.abs().mean().item()
    print(f"  mean |out(P_a) - out(P_b)| : {delta:.6f}")
    print(f"  mean |out|                 : {scale:.6f}")
    print(f"  relative change            : {delta / max(scale, 1e-9) * 100:.2f}%")

    ok = delta > 1e-4
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def check_head_count(device: str) -> bool:
    print("=" * 72)
    print("CHECK 3 — num_heads changes the computation")
    print("=" * 72)

    B, N, C = 2, 64, 128
    torch.manual_seed(2)
    tokens = torch.randn(B, N, C, device=device)
    physics = torch.randn(B, 64, 32, 32, device=device)

    torch.manual_seed(3)
    a1 = PhysicsGuidedAttention(embed_dim=C, num_heads=1).to(device)
    torch.manual_seed(3)
    a4 = PhysicsGuidedAttention(embed_dim=C, num_heads=4).to(device)

    # Copy the shared QKV weights so ONLY the head split differs.
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            getattr(a4, name).weight.copy_(getattr(a1, name).weight)
            getattr(a4, name).bias.copy_(getattr(a1, name).bias)
        o1 = a1(tokens, tokens, tokens, physics)
        o4 = a4(tokens, tokens, tokens, physics)

    delta = (o1 - o4).abs().mean().item()
    print(f"  num_heads=1 vs num_heads=4, identical QKV weights")
    print(f"  mean |difference| : {delta:.6f}")
    ok = delta > 1e-5
    print(f"  RESULT: {'PASS' if ok else 'FAIL'} (a real head split must change the result)")
    return ok


def check_full_model(device: str) -> bool:
    print("=" * 72)
    print("CHECK 4 — full model forward + backward at training shape")
    print("=" * 72)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model = build_model(device=device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable parameters : {n_params:,}")

    # Mirror the real training step exactly: inputs in [0,1], an L1 objective, AMP
    # autocast AND a GradScaler. The scaler is essential — this model's gradients at the
    # attention projections are ~1e-4, which underflows to exactly zero in raw fp16.
    x = torch.rand(4, 3, 256, 256, device=device)
    target = torch.rand(4, 3, 256, 256, device=device)
    scaler = torch.amp.GradScaler(device=device, enabled=(device == "cuda"))
    with torch.autocast(device_type=device, enabled=(device == "cuda")):
        y = model(x)
        loss = F.l1_loss(y, target)
    scaler.scale(loss).backward()
    scaler.unscale_(torch.optim.SGD(model.parameters(), lr=0.0))

    finite = torch.isfinite(y).all().item()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    grads_finite = all(torch.isfinite(g).all().item() for g in grads)

    # Confirm the new projections exist and receive gradient.
    qw = model.low_freq_transformer.attn.q_proj.weight
    vw = model.low_freq_transformer.attn.v_proj.weight
    print(f"  output shape         : {tuple(y.shape)}")
    print(f"  output finite        : {finite}")
    print(f"  all grads finite     : {grads_finite}")
    print(f"  q_proj grad norm     : {qw.grad.norm().item():.6e}")
    print(f"  v_proj grad norm     : {vw.grad.norm().item():.6e}")
    print(f"  physics_scale grad   : {model.low_freq_transformer.attn.physics_scale.grad.item():.6e}")
    if device == "cuda":
        print(f"  peak VRAM            : {torch.cuda.max_memory_allocated() / 1e9:.2f} GB (batch=4, 256x256)")

    ok = (
        finite and grads_finite
        and qw.grad.norm().item() > 0
        and vw.grad.norm().item() > 0
        and tuple(y.shape) == (4, 3, 256, 256)
    )
    print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")
    results = [
        ("convex hull", check_convex_hull(device)),
        ("physics sensitivity", check_physics_sensitivity(device)),
        ("head count", check_head_count(device)),
        ("full model", check_full_model(device)),
    ]
    print("=" * 72)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("=" * 72)
    sys.exit(0 if all(ok for _, ok in results) else 1)


if __name__ == "__main__":
    main()
