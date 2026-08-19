"""Phase-0 gate: 5 real training steps on real UIEB data, unmodified model.

Proves the environment is sound (data -> model -> loss -> backward -> step)
before any code is edited.
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset

from data.dataset import UIEBDataset
from models.model import PFGTUIEModel
from models.loss import PFGTLoss

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = UIEBDataset(image_size=256, augment=True)
    print(f"dataset pairs: {len(ds)}")
    loader = DataLoader(Subset(ds, list(range(40))), batch_size=8, shuffle=True, num_workers=0)

    model = PFGTUIEModel(embed_dim=128, num_heads=4).to(dev)
    crit = PFGTLoss().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scaler = GradScaler(device=dev, enabled=(dev == "cuda"))

    model.train()
    losses = []
    step = 0
    t0 = time.time()
    for epoch in range(2):
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with autocast(device_type=dev, enabled=(dev == "cuda")):
                out = model(x)
                l = crit(out, y)
            tot = l["total_loss"]
            if not torch.isfinite(tot):
                print("FAIL: non-finite loss"); sys.exit(1)
            scaler.scale(tot).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            losses.append(float(tot))
            print(f"step {step}: loss={float(tot):.6f} l1={float(l['l1_loss']):.6f} "
                  f"ssim={float(l['ssim_loss']):.6f} perc={float(l['perceptual_loss']):.6f} gn={float(gn):.3f}")
            step += 1
            if step >= 5:
                break
        if step >= 5:
            break

    peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
    print(f"\n5 steps in {time.time()-t0:.1f}s   peak VRAM {peak:.2f} GB")
    print(f"loss first={losses[0]:.6f} last={losses[-1]:.6f}  delta={losses[-1]-losses[0]:+.6f}")
    print("GATE PASS: env sound (data+model+loss+backward+optimizer all ran on %s)" % dev)

if __name__ == "__main__":
    main()
