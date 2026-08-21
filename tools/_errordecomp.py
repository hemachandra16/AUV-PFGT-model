"""Where does the model's remaining error actually live: low frequency or high frequency?

65% of the model's parameters (1.77M of 2.73M) sit in the high-frequency transformer, and
only 7.3% in the low-frequency one. That is only a sensible split if the residual error is
dominated by high-frequency detail. Underwater enhancement is mostly a colour/illumination
problem, which lives in the LL band -- so this measures which it actually is.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import get_splits
from models.build import build_model
from models.wavelet import WaveletTransform
from utils.checkpoint import load_checkpoint

dev='cuda'
_, val = get_splits(augment_train=False)
loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0)
m = build_model(device=dev); load_checkpoint('checkpoints/best.pt', model=m, device=torch.device(dev)); m.eval()
wt = WaveletTransform().to(dev)

acc = np.zeros(4); accE = np.zeros(4); n=0
glob = 0.0; tot_err = 0.0
with torch.no_grad():
    for x,y in loader:
        x,y = x.to(dev), y.to(dev)
        out = m(x)
        err = out - y
        eb = wt(err.float())                      # error decomposed into wavelet bands
        yb = wt(y.float())
        for j in range(4):
            acc[j]  += float((eb[j]**2).mean())   # error energy per band
            accE[j] += float((yb[j]**2).mean())   # target energy per band
        # how much of the error is a pure global per-channel offset (i.e. colour cast)?
        gmean = err.mean(dim=(2,3), keepdim=True)
        glob += float((gmean**2).mean() * err[0,0].numel()) / err[0,0].numel()
        glob_energy = float((gmean.expand_as(err)**2).mean())
        tot_err += float((err**2).mean())
        n+=1
acc/=n; accE/=n
names=['LL (colour/illumination)','LH (horizontal detail)','HL (vertical detail)','HH (diagonal detail)']
print(f"held-out, {n} batches\n")
print(f"{'band':<28}{'error energy':>14}{'% of error':>12}{'target energy':>15}")
print('-'*72)
for j,nm in enumerate(names):
    print(f"{nm:<28}{acc[j]:>14.6f}{acc[j]/acc.sum()*100:>11.1f}%{accE[j]:>15.6f}")
print('-'*72)
print(f"{'TOTAL':<28}{acc.sum():>14.6f}")
print(f"\nlow-frequency (LL) share of remaining error : {acc[0]/acc.sum()*100:.1f}%")
print(f"high-frequency (LH+HL+HH) share            : {acc[1:].sum()/acc.sum()*100:.1f}%")
print(f"\nparameter allocation:")
lo = sum(p.numel() for p in m.low_freq_transformer.parameters())
hi = sum(p.numel() for p in m.high_freq_transformer.parameters())
tp = sum(p.numel() for p in m.parameters())
print(f"  low_freq_transformer  : {lo:>9,}  ({lo/tp*100:.1f}%)")
print(f"  high_freq_transformer : {hi:>9,}  ({hi/tp*100:.1f}%)")
print(f"  ratio high:low        : {hi/lo:.1f}x")
