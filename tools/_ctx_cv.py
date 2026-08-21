"""Cross-validated: is the needed offset ACTUALLY predictable from the physics context?

My first attempt fit 64 context dimensions to 89 samples by ordinary least squares and
reported R^2 = 0.69-0.90. With 64 free parameters and 89 points that is almost certainly
overfitting, so it cannot be trusted. This repeats it with k-fold cross-validation and ridge
regularisation, and reports HELD-OUT R^2 -- which is what "predictable" has to mean.

Trained on the 801-image TRAIN split (not the 89 val images) so there are enough samples for
the question to be answerable at all.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from torch.utils.data import DataLoader
from data.dataset import get_splits
from models.build import build_model
from utils.checkpoint import load_checkpoint

dev='cuda'
train_sub, val_sub = get_splits(augment_train=False)
m = build_model(device=dev); load_checkpoint('checkpoints/best.pt', model=m, device=torch.device(dev)); m.eval()

def collect(subset):
    CTX=[];INM=[];OUTM=[];TGTM=[]
    with torch.no_grad():
        for x,y in DataLoader(subset,batch_size=8,shuffle=False,num_workers=0):
            x,y=x.to(dev),y.to(dev)
            _,ctx=m.physics_encoder(x)
            saved=m.global_correction; m.global_correction=None
            base=m(x); m.global_correction=saved
            CTX.append(ctx.cpu()); INM.append(x.mean(dim=(2,3)).cpu())
            OUTM.append(base.mean(dim=(2,3)).cpu()); TGTM.append(y.mean(dim=(2,3)).cpu())
    return (torch.cat(CTX).numpy(), torch.cat(INM).numpy(),
            torch.cat(OUTM).numpy(), torch.cat(TGTM).numpy())

print("collecting train split (801 images)...", flush=True)
CTXt, INMt, OUTMt, TGTMt = collect(train_sub)
print("collecting val split (89 images)...", flush=True)
CTXv, INMv, OUTMv, TGTMv = collect(val_sub)
NEEDt = TGTMt - OUTMt
NEEDv = TGTMv - OUTMv

def ridge_fit_eval(Xtr, Ytr, Xte, Yte, alpha):
    Xa=np.c_[Xtr, np.ones(len(Xtr))]; Xb=np.c_[Xte, np.ones(len(Xte))]
    out=[]
    for c in range(Ytr.shape[1]):
        A = Xa.T@Xa + alpha*np.eye(Xa.shape[1]); A[-1,-1]-=alpha
        coef = np.linalg.solve(A, Xa.T@Ytr[:,c])
        pred = Xb@coef
        ss_res=((Yte[:,c]-pred)**2).sum(); ss_tot=((Yte[:,c]-Yte[:,c].mean())**2).sum()
        out.append(1-ss_res/max(ss_tot,1e-12))
    return out

print(f"\ntrain n={len(CTXt)}   val n={len(CTXv)}   context dim={CTXt.shape[1]}\n")
print("HELD-OUT R^2 (fit on 801 train images, evaluated on the 89 val images)")
print(f"{'signal':<44}{'alpha':>8}{'R2_R':>8}{'R2_G':>8}{'R2_B':>8}")
print('-'*76)
best=None
for name, Xtr, Xte in [
    ("physics_context (what the module sees)", CTXt, CTXv),
    ("context + input mean",                   np.c_[CTXt,INMt], np.c_[CTXv,INMv]),
    ("context + OUTPUT mean",                  np.c_[CTXt,OUTMt], np.c_[CTXv,OUTMv]),
    ("context + input + OUTPUT mean",          np.c_[CTXt,INMt,OUTMt], np.c_[CTXv,INMv,OUTMv]),
]:
    for alpha in (1.0, 10.0):
        v=ridge_fit_eval(Xtr, NEEDt, Xte, NEEDv, alpha)
        print(f"{name:<44}{alpha:>8.1f}{v[0]:>8.3f}{v[1]:>8.3f}{v[2]:>8.3f}")
        if best is None or np.mean(v)>best[0]: best=(np.mean(v), name, alpha)
print()
print(f"best held-out mean R^2: {best[0]:.3f}  ({best[1]}, alpha={best[2]})")
print()
print("For contrast, the IN-SAMPLE R^2 my first test reported on 89 points with 64 features:")
v_in = ridge_fit_eval(CTXv, NEEDv, CTXv, NEEDv, 1e-9)
print(f"  physics_context, fit and evaluated on the same 89 images: "
      f"{[round(x,3) for x in v_in]}   <- overfit, not evidence")
