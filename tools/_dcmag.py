"""Measure the magnitude of a candidate L_dc term, to choose lambda_dc sensibly.

lambda_dc has to be strong enough to actually move GlobalColorCorrection -- it competes with
L1 (weight 1.0), SSIM (0.5) and perceptual (0.1), all of which are dominated by spatial error.
Too weak reproduces session 3's failure; too strong trades everything else for colour.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import get_splits
from models.build import build_model
from models.loss import PFGTLoss
from utils.checkpoint import load_checkpoint

dev='cuda'
_, val = get_splits(augment_train=False)
loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0)
m = build_model(device=dev); load_checkpoint('checkpoints/best.pt', model=m, device=torch.device(dev)); m.eval()
crit = PFGTLoss(lambda_l1=1.0, lambda_ssim=0.5, lambda_perceptual=0.1, lambda_frequency=0.0).to(dev)

acc={'l1':0.,'ssim':0.,'perc':0.,'dc_l1':0.,'dc_mse':0.}; n=0
with torch.no_grad():
    for x,y in loader:
        x,y=x.to(dev),y.to(dev)
        out=m(x); l=crit(out,y)
        pm=out.mean(dim=(2,3)); tm=y.mean(dim=(2,3))
        acc['l1']+=float(l['l1_loss']); acc['ssim']+=float(l['ssim_loss']); acc['perc']+=float(l['perceptual_loss'])
        acc['dc_l1']+=float((pm-tm).abs().mean()); acc['dc_mse']+=float(((pm-tm)**2).mean())
        n+=1
for k in acc: acc[k]/=n

W={'l1':1.0,'ssim':0.5,'perc':0.1}
print(f"held-out, {n} batches\n")
print(f"{'term':<12}{'raw':>10}{'weight':>9}{'weighted':>11}")
print('-'*44)
tot=sum(W[k]*acc[k] for k in W)
for k in W: print(f"{k:<12}{acc[k]:>10.5f}{W[k]:>9.2f}{W[k]*acc[k]:>11.5f}")
print('-'*44); print(f"{'TOTAL':<12}{'':>10}{'':>9}{tot:>11.5f}\n")
print(f"candidate L_dc (L1 on per-image per-channel means) : {acc['dc_l1']:.5f}")
print(f"candidate L_dc (MSE)                               : {acc['dc_mse']:.6f}")
print()
print("gradient-strength argument for L1 over MSE:")
print(f"  DC errors are small (~{acc['dc_l1']:.3f}). d/dx of |x| is 1.0; d/dx of x^2 is 2x = {2*acc['dc_l1']:.3f}.")
print(f"  So MSE would give a ~{1/(2*acc['dc_l1']):.0f}x WEAKER push exactly where we need it.")
print()
for lam in (0.1,0.2,0.3,0.5):
    w=lam*acc['dc_l1']
    print(f"  lambda_dc={lam:<4} -> weighted {w:.5f} = {w/(tot+w)*100:5.1f}% of the total objective")
