"""Do closed-form physics priors carry information the current encoder cannot easily get?

The current PhysicsPriorEncoder is Conv3x3(3->64) -> GELU -> 2x ResBlock -> Conv3x3(64->64).
Its ONLY input is the same RGB the wavelet branch already sees. So the question is whether
cheap, closed-form underwater priors carry signal about the correction actually required --
signal a small local conv stack over RGB would struggle to produce (most of these priors are
GLOBAL or min-over-channel statistics, which 3x3 convs cannot compute).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import get_splits

def priors(x):
    """x: (B,3,H,W) in [0,1]. Cheap closed-form underwater priors."""
    r,g,b = x[:,0:1], x[:,1:2], x[:,2:3]
    dcp = -F.max_pool2d(-x.min(dim=1,keepdim=True).values, 15, 1, 7)   # dark channel
    bcp =  F.max_pool2d( x.max(dim=1,keepdim=True).values, 15, 1, 7)   # bright channel
    mu  = x.mean(dim=(2,3), keepdim=True)                              # per-channel mean
    # red attenuation: how depleted red is vs the max channel (the dominant underwater cue)
    ratio_rg = (mu[:,0:1]/(mu[:,1:2]+1e-6))
    ratio_rb = (mu[:,0:1]/(mu[:,2:3]+1e-6))
    localstd = (F.avg_pool2d(x**2,15,1,7) - F.avg_pool2d(x,15,1,7)**2).clamp_min(0).sqrt().mean(1,keepdim=True)
    return dict(dcp=dcp, bcp=bcp, localstd=localstd,
                mu_r=mu[:,0,0,0], mu_g=mu[:,1,0,0], mu_b=mu[:,2,0,0],
                ratio_rg=ratio_rg[:,0,0,0], ratio_rb=ratio_rb[:,0,0,0])

_, val = get_splits(augment_train=False)
tr, _  = get_splits(augment_train=False)
loader = DataLoader(tr, batch_size=16, shuffle=False, num_workers=0)

feats=[]; targets=[]
with torch.no_grad():
    for i,(x,y) in enumerate(loader):
        if i>=25: break
        p = priors(x)
        # target = the per-channel gain the reference actually applies to the raw image
        gain = (y.mean(dim=(2,3)) + 1e-6) / (x.mean(dim=(2,3)) + 1e-6)   # (B,3)
        f = torch.stack([p['dcp'].mean((1,2,3)), p['bcp'].mean((1,2,3)),
                         p['localstd'].mean((1,2,3)),
                         p['mu_r'], p['mu_g'], p['mu_b'],
                         p['ratio_rg'], p['ratio_rb']], dim=1)
        feats.append(f); targets.append(gain)
X = torch.cat(feats).numpy(); Y = torch.cat(targets).numpy()
print(f"samples: {X.shape[0]}   prior features: {X.shape[1]}   targets: per-channel gain (R,G,B)\n")

names=['dcp','bcp','local_std','mu_R','mu_G','mu_B','R/G','R/B']
print("correlation of each closed-form prior with the colour correction actually needed:")
print(f"{'prior':<12}{'gain_R':>10}{'gain_G':>10}{'gain_B':>10}")
print('-'*44)
for j,nm in enumerate(names):
    cs=[np.corrcoef(X[:,j],Y[:,c])[0,1] for c in range(3)]
    print(f"{nm:<12}{cs[0]:>10.3f}{cs[1]:>10.3f}{cs[2]:>10.3f}")

# How much of the required gain is linearly predictable from these 8 numbers alone?
Xa = np.c_[X, np.ones(len(X))]
print("\nlinear predictability of the needed gain from the 8 priors (R^2):")
for c,ch in enumerate('RGB'):
    coef,_,_,_ = np.linalg.lstsq(Xa, Y[:,c], rcond=None)
    pred = Xa@coef
    ss_res=((Y[:,c]-pred)**2).sum(); ss_tot=((Y[:,c]-Y[:,c].mean())**2).sum()
    print(f"  gain_{ch}: R^2 = {1-ss_res/ss_tot:.3f}")
print("\nNote: mu_*, R/G, R/B and the pooled dcp/bcp are GLOBAL statistics.")
print("A stack of 3x3 convs has a small receptive field and no global pooling,")
print("so it cannot compute these directly however many layers it has.")
