"""Is the needed per-image offset predictable from what GlobalColorCorrection can SEE?

The module predicts an affine from physics_context, which is computed from the INPUT image.
But the offset actually needed is  mean(target) - mean(model_output)  -- and model_output is
the result of the whole network's processing, which the context never observes.

So: regress each candidate signal against the needed offset and compare R^2.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from torch.utils.data import DataLoader
from data.dataset import get_splits
from models.build import build_model
from utils.checkpoint import load_checkpoint

dev='cuda'
_, val = get_splits(augment_train=False)
loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0)
m = build_model(device=dev); load_checkpoint('checkpoints/best.pt', model=m, device=torch.device(dev)); m.eval()

CTX=[]; INMEAN=[]; OUTMEAN=[]; TGTMEAN=[]
with torch.no_grad():
    for x,y in loader:
        x,y=x.to(dev),y.to(dev)
        _,ctx = m.physics_encoder(x)
        saved=m.global_correction; m.global_correction=None
        base=m(x); m.global_correction=saved
        CTX.append(ctx.cpu()); INMEAN.append(x.mean(dim=(2,3)).cpu())
        OUTMEAN.append(base.mean(dim=(2,3)).cpu()); TGTMEAN.append(y.mean(dim=(2,3)).cpu())
CTX=torch.cat(CTX).numpy(); INMEAN=torch.cat(INMEAN).numpy()
OUTMEAN=torch.cat(OUTMEAN).numpy(); TGTMEAN=torch.cat(TGTMEAN).numpy()
NEEDED = TGTMEAN - OUTMEAN                      # what the module must output as `shift`

def r2(X, Y):
    Xa=np.c_[X, np.ones(len(X))]
    out=[]
    for c in range(Y.shape[1]):
        coef,_,_,_=np.linalg.lstsq(Xa, Y[:,c], rcond=None)
        pred=Xa@coef
        ss_res=((Y[:,c]-pred)**2).sum(); ss_tot=((Y[:,c]-Y[:,c].mean())**2).sum()
        out.append(1-ss_res/max(ss_tot,1e-12))
    return out

print(f"samples: {len(CTX)}   context dim: {CTX.shape[1]}\n")
print("How well can each available signal predict the offset the module must apply?")
print(f"{'signal available to the module':<46}{'R2_R':>8}{'R2_G':>8}{'R2_B':>8}")
print('-'*70)
for name, X in [
    ("physics_context ONLY  (what it gets today)", CTX),
    ("input per-channel mean only",                INMEAN),
    ("context + input mean",                       np.c_[CTX, INMEAN]),
    ("OUTPUT per-channel mean only",               OUTMEAN),
    ("context + OUTPUT mean",                      np.c_[CTX, OUTMEAN]),
    ("context + input mean + OUTPUT mean",         np.c_[CTX, INMEAN, OUTMEAN]),
]:
    v=r2(X, NEEDED)
    print(f"{name:<46}{v[0]:>8.3f}{v[1]:>8.3f}{v[2]:>8.3f}")

print()
print("And for reference, predicting the TARGET mean itself (an absolute, not a correction):")
print(f"{'signal':<46}{'R2_R':>8}{'R2_G':>8}{'R2_B':>8}")
print('-'*70)
for name, X in [("physics_context ONLY", CTX), ("context + input mean", np.c_[CTX, INMEAN])]:
    v=r2(X, TGTMEAN)
    print(f"{name:<46}{v[0]:>8.3f}{v[1]:>8.3f}{v[2]:>8.3f}")
print()
print("Interpretation: the module is asked for a CORRECTION relative to what the network")
print("already produced, but it never observes what the network produced.")
