"""Verify the claim: how much PSNR is available from a per-image per-channel offset alone?

Two independent reviewers converged on the same story -- that the model's dominant remaining
error is a per-image global colour offset it structurally cannot emit. This checks the
headline number directly rather than trusting it.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from torch.utils.data import DataLoader
from data.dataset import get_splits
from models.build import build_model
from utils.checkpoint import load_checkpoint
from metrics import compute_psnr, compute_ssim

dev='cuda'
_, val = get_splits(augment_train=False)
loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0)
m = build_model(device=dev); load_checkpoint('checkpoints/best.pt', model=m, device=torch.device(dev)); m.eval()

base_p=base_s=off_p=off_s=aff_p=aff_s=0.0; n=0
dc_share=0.0
with torch.no_grad():
    for x,y in loader:
        x,y = x.to(dev), y.to(dev)
        out = m(x)
        for i in range(out.shape[0]):
            p,t = out[i:i+1], y[i:i+1]
            base_p += compute_psnr(p,t); base_s += compute_ssim(p,t)
            # oracle per-image per-channel OFFSET
            d = (t-p).mean(dim=(2,3), keepdim=True)
            po = (p+d).clamp(0,1)
            off_p += compute_psnr(po,t); off_s += compute_ssim(po,t)
            # oracle per-image per-channel AFFINE (gain+offset), least squares per channel
            pa = p.clone()
            for c in range(3):
                a = p[0,c].flatten(); b = t[0,c].flatten()
                A = torch.stack([a, torch.ones_like(a)],1)
                sol = torch.linalg.lstsq(A, b.unsqueeze(1)).solution
                pa[0,c] = (a*sol[0,0]+sol[1,0]).reshape(p.shape[2],p.shape[3])
            pa = pa.clamp(0,1)
            aff_p += compute_psnr(pa,t); aff_s += compute_ssim(pa,t)
            # fraction of error energy that is pure DC
            e = p-t
            dc = e.mean(dim=(2,3), keepdim=True)
            dc_share += float((dc**2).mean()/ (e**2).mean())
            n+=1
print(f"held-out {n} images\n")
print(f"{'condition':<44}{'PSNR':>9}{'SSIM':>9}{'delta':>9}")
print('-'*72)
print(f"{'model as-is':<44}{base_p/n:>9.4f}{base_s/n:>9.4f}{0.0:>9.3f}")
print(f"{'+ oracle per-image per-channel OFFSET':<44}{off_p/n:>9.4f}{off_s/n:>9.4f}{off_p/n-base_p/n:>+9.3f}")
print(f"{'+ oracle per-image per-channel AFFINE':<44}{aff_p/n:>9.4f}{aff_s/n:>9.4f}{aff_p/n-base_p/n:>+9.3f}")
print(f"\nfraction of error energy that is pure per-image per-channel DC: {dc_share/n*100:.1f}%")
print(f"\nfor scale: the gap to the pre-fix baseline is 25.114 - 24.902 = 0.212 dB")
