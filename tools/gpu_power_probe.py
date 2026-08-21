"""Measure whether the GPU is being power-capped under SUSTAINED load.

Last night the RTX 4060 ran the first ~2 epochs at full speed and then clamped to
19.46 W / 585 MHz with `clocks_event_reasons.active = 0x4` (SW Power Cap). A short burst
therefore proves nothing — the cap only engages after sustained draw. This runs a real
PFGT-UIE training step in a loop for a fixed duration and samples nvidia-smi throughout,
so the steady-state behaviour is what gets measured.

Reference (throttled, Windows "Balanced" power plan, 2026-08-20):
    power.draw 19.46 W | clocks.sm 585 MHz | temp 52 C | reasons 0x4 (SW Power Cap)

Throttle reason bits: 0x1 GpuIdle, 0x2 AppClocksSetting, 0x4 SwPowerCap,
0x8 HwSlowdown, 0x20 SwThermalSlowdown, 0x40 HwThermalSlowdown, 0x80 HwPowerBrake.

Usage:
    python tools/gpu_power_probe.py --seconds 120
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.build import build_model
from models.loss import PFGTLoss

REASONS = [
    (0x1, "GpuIdle"), (0x2, "AppClocksSetting"), (0x4, "SwPowerCap"),
    (0x8, "HwSlowdown"), (0x10, "SyncBoost"), (0x20, "SwThermalSlowdown"),
    (0x40, "HwThermalSlowdown"), (0x80, "HwPowerBrake"),
]


def decode(mask_hex: str) -> str:
    try:
        m = int(mask_hex, 16)
    except ValueError:
        return mask_hex
    if m == 0:
        return "none"
    hits = [name for bit, name in REASONS if m & bit]
    return "+".join(hits) if hits else hex(m)


def sample() -> tuple:
    out = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=power.draw,clocks.sm,clocks.max.sm,temperature.gpu,"
         "utilization.gpu,clocks_event_reasons.active",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=20,
    ).stdout.strip()
    parts = [p.strip() for p in out.split(",")]
    return parts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--interval", type=int, default=10)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")

    dev = "cuda"
    model = build_model(device=dev)
    crit = PFGTLoss().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = torch.amp.GradScaler(device=dev)
    model.train()

    x = torch.rand(args.batch_size, 3, 256, 256, device=dev)
    y = torch.rand(args.batch_size, 3, 256, 256, device=dev)

    print(f"Sustained load: {args.seconds}s of real PFGT-UIE training steps "
          f"(batch {args.batch_size}, 256x256)\n")
    print(f"{'t(s)':>5} {'power(W)':>9} {'sm(MHz)':>8} {'max':>6} {'%max':>6} "
          f"{'temp':>5} {'util%':>6}  throttle_reason")
    print("-" * 78)

    t0 = time.time()
    next_report = 0.0
    steps = 0
    samples = []
    while time.time() - t0 < args.seconds:
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda"):
            out = model(x)
            loss = crit(out, y)
        scaler.scale(loss["total_loss"]).backward()
        scaler.step(opt)
        scaler.update()
        steps += 1

        el = time.time() - t0
        if el >= next_report:
            torch.cuda.synchronize()
            p = sample()
            try:
                pw, sm, mx, tp, ut, rs = float(p[0]), int(p[1]), int(p[2]), p[3], p[4], p[5]
                pct = sm / max(mx, 1) * 100
                print(f"{el:5.0f} {pw:9.2f} {sm:8d} {mx:6d} {pct:5.1f}% {tp:>5} {ut:>6}  {decode(rs)}")
                samples.append((pw, sm, pct, decode(rs)))
            except (ValueError, IndexError):
                print(f"{el:5.0f}  (unparsed nvidia-smi output: {p})")
            next_report = el + args.interval

    dt = time.time() - t0
    print("-" * 78)
    print(f"steps: {steps} in {dt:.0f}s -> {steps/dt:.2f} steps/s "
          f"({dt/max(steps,1)*100:.0f}s per 100-step epoch)")

    if samples:
        # Ignore the first sample: the cap takes a few seconds to engage.
        steady = samples[1:] or samples
        avg_pw = sum(s[0] for s in steady) / len(steady)
        avg_sm = sum(s[1] for s in steady) / len(steady)
        avg_pct = sum(s[2] for s in steady) / len(steady)
        capped = sum(1 for s in steady if "SwPowerCap" in s[3])
        print(f"\nSTEADY STATE (excluding first sample):")
        print(f"  mean power : {avg_pw:.2f} W      (throttled reference: 19.46 W)")
        print(f"  mean clock : {avg_sm:.0f} MHz = {avg_pct:.1f}% of max   "
              f"(throttled reference: 585 MHz = 18.8%)")
        print(f"  SwPowerCap active in {capped}/{len(steady)} samples")
        print()
        # NOTE on interpreting SwPowerCap: a GPU running flat out at its board TDP
        # reports SwPowerCap essentially all the time — it means "currently limited by
        # the power budget", which is the normal, healthy state at full load. The flag
        # alone therefore says nothing. What distinguishes a *reduced* cap from the
        # normal one is the magnitude: last night's clamp sat at 19 W / 585 MHz (19% of
        # max clock); an unclamped RTX 4060 Laptop sits near its ~77 W board limit at
        # 2400-2600 MHz (~80% of max). So judge on power and clock, not on the flag.
        pct_of_board = avg_pw / 77.0 * 100
        if avg_pw > 45 and avg_pct > 55:
            print(f"  VERDICT: throttle LIFTED. {avg_pw:.0f} W is {pct_of_board:.0f}% of the "
                  f"77 W board limit and the clock sits at {avg_pct:.0f}% of max.")
            print(f"           SwPowerCap in {capped}/{len(steady)} samples is EXPECTED here — "
                  f"it just means the card is running at its full power budget.")
        elif avg_pw < 30 or avg_pct < 35:
            print(f"  VERDICT: STILL THROTTLED — {avg_pw:.0f} W / {avg_pct:.0f}% of max clock is "
                  f"close to the 19 W / 19% clamp. Budget ~146 s/epoch.")
        else:
            print("  VERDICT: PARTIAL / intermittent capping — see samples above.")

        hot = [s for s in steady if "ThermalSlowdown" in s[3]]
        if hot:
            print(f"  WARNING: thermal slowdown seen in {len(hot)} samples — sustained runs "
                  f"may lose clock as the card heats.")


if __name__ == "__main__":
    main()
