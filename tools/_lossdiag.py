"""Measure what each loss term ACTUALLY contributes, on the trained model + real data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from data.dataset import get_splits
from models.build import build_model
from models.loss import PFGTLoss
from utils.checkpoint import load_checkpoint

dev='cuda'
_, val = get_splits(augment_train=False)
loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0)
m = build_model(device=dev); load_checkpoint('checkpoints/best.pt', model=m, device=torch.device(dev)); m.eval()
crit = PFGTLoss(lambda_l1=1.0, lambda_ssim=0.5, lambda_perceptual=0.1, lambda_frequency=0.15).to(dev)

acc = {k:0.0 for k in ['l1_loss','ssim_loss','perceptual_loss','frequency_loss','total_loss']}
n=0
with torch.no_grad():
    for x,y in loader:
        x,y = x.to(dev), y.to(dev)
        out = m(x); l = crit(out,y)
        for k in acc: acc[k] += float(l[k])
        n+=1
for k in acc: acc[k] /= n

W = {'l1_loss':1.0,'ssim_loss':0.5,'perceptual_loss':0.1,'frequency_loss':0.15}
print(f"held-out {n} batches\n")
print(f"{'term':<18}{'raw':>10}{'weight':>9}{'weighted':>11}{'% of total':>12}")
print('-'*60)
tot = sum(W[k]*acc[k] for k in W)
for k in W:
    w = W[k]*acc[k]
    print(f"{k:<18}{acc[k]:>10.5f}{W[k]:>9.2f}{w:>11.5f}{w/tot*100:>11.1f}%")
print('-'*60)
print(f"{'TOTAL':<18}{'':>10}{'':>9}{tot:>11.5f}")

# Is the frequency loss redundant with L1? Haar LL is a local average -> correlated with L1.
print("\n--- redundancy check: correlation of per-sample L1 vs each frequency sub-band L1 ---")
import torch.nn.functional as F
from models.wavelet import WaveletTransform
wt = WaveletTransform().to(dev)
rows=[]
with torch.no_grad():
    for x,y in loader:
        x,y = x.to(dev), y.to(dev)
        out = m(x)
        for i in range(out.shape[0]):
            p,t = out[i:i+1], y[i:i+1]
            l1 = float(F.l1_loss(p,t))
            pb = wt(p.float()); tb = wt(t.float())
            bands = [float(F.l1_loss(a,b)) for a,b in zip(pb,tb)]
            rows.append([l1]+bands)
import numpy as np
A=np.array(rows)
names=['LL','LH','HL','HH']
for j,nm in enumerate(names):
    r=np.corrcoef(A[:,0],A[:,1+j])[0,1]
    print(f"  corr(pixel L1, {nm} band L1) = {r:+.4f}   mean {A[:,1+j].mean():.5f}")
print(f"  corr(pixel L1, mean-of-4-bands) = {np.corrcoef(A[:,0],A[:,1:].mean(1))[0,1]:+.4f}")
