"""How much of the +3.36 dB oracle headroom is ACHIEVABLE, not just theoretical?

tools/_oracle_dc.py computes its offset as (target - prediction).mean(...) -- it uses the
GROUND TRUTH. That makes it an upper bound, not a target a model can reach. A model sees only
the input.

Held-out ridge regression says the needed offset is only weakly predictable from the physics
context: R^2 = 0.015 / 0.104 / 0.346 (mean 0.155). So this fits the best linear predictor on
the 801 TRAIN images and applies it to the 89 held-out ones, measuring the PSNR it actually
buys. That is the realistic ceiling for any input-conditioned global colour correction.
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
train_sub, val_sub = get_splits(augment_train=False)
m = build_model(device=dev); load_checkpoint('checkpoints/best.pt', model=m, device=torch.device(dev)); m.eval()

def collect(subset, keep_images=False):
    CTX=[];OUT=[];TGT=[];IMGS=[]
    with torch.no_grad():
        for x,y in DataLoader(subset,batch_size=8,shuffle=False,num_workers=0):
            x,y=x.to(dev),y.to(dev)
            _,ctx=m.physics_encoder(x)
            saved=m.global_correction; m.global_correction=None
            base=m(x); m.global_correction=saved
            CTX.append(ctx.cpu()); OUT.append(base.mean(dim=(2,3)).cpu()); TGT.append(y.mean(dim=(2,3)).cpu())
            if keep_images: IMGS.append((base.cpu(), y.cpu()))
    return torch.cat(CTX).numpy(), torch.cat(OUT).numpy(), torch.cat(TGT).numpy(), IMGS

CTXt,OUTt,TGTt,_ = collect(train_sub)
CTXv,OUTv,TGTv,IMGS = collect(val_sub, keep_images=True)
NEEDt = TGTt-OUTt

# best linear predictor of the needed offset, fit on TRAIN only
alpha=1.0
Xa=np.c_[CTXt,np.ones(len(CTXt))]
coefs=[]
for c in range(3):
    A=Xa.T@Xa+alpha*np.eye(Xa.shape[1]); A[-1,-1]-=alpha
    coefs.append(np.linalg.solve(A, Xa.T@NEEDt[:,c]))
coefs=np.stack(coefs,1)
Xb=np.c_[CTXv,np.ones(len(CTXv))]
pred_off = torch.from_numpy(Xb@coefs).float()          # (N,3) predicted offsets

base_p=base_s=pred_p=pred_s=orac_p=orac_s=cmean_p=0.0; n=0
i=0
for base,y in IMGS:
    for b in range(base.shape[0]):
        p=base[b:b+1]; t=y[b:b+1]
        base_p+=compute_psnr(p,t); base_s+=compute_ssim(p,t)
        # oracle (uses ground truth)
        d=(t-p).mean(dim=(2,3),keepdim=True)
        orac_p+=compute_psnr((p+d).clamp(0,1),t); orac_s+=compute_ssim((p+d).clamp(0,1),t)
        # realistic: offset predicted from the INPUT only
        po=pred_off[i].view(1,3,1,1)
        pp=(p+po).clamp(0,1)
        pred_p+=compute_psnr(pp,t); pred_s+=compute_ssim(pp,t)
        # constant offset (dataset-wide mean) -- the "near-constant" strategy session 3 learned
        cm=torch.from_numpy((TGTt-OUTt).mean(0)).float().view(1,3,1,1)
        cmean_p+=compute_psnr((p+cm).clamp(0,1),t)
        i+=1; n+=1

print(f"held-out {n} images\n")
print(f"{'condition':<52}{'PSNR':>9}{'delta':>9}")
print('-'*72)
print(f"{'model as-is (global correction removed)':<52}{base_p/n:>9.4f}{0.0:>9.3f}")
print(f"{'+ CONSTANT offset (dataset mean) - what S3 learned':<52}{cmean_p/n:>9.4f}{cmean_p/n-base_p/n:>+9.3f}")
print(f"{'+ offset PREDICTED from input (best linear, held-out)':<52}{pred_p/n:>9.4f}{pred_p/n-base_p/n:>+9.3f}")
print(f"{'+ ORACLE offset (uses ground truth) - upper bound':<52}{orac_p/n:>9.4f}{orac_p/n-base_p/n:>+9.3f}")
print('-'*72)
ach=(pred_p/n-base_p/n); orc=(orac_p/n-base_p/n)
print(f"\nachievable / oracle = {ach:.3f} / {orc:.3f} = {ach/max(orc,1e-9)*100:.1f}% of the headroom")
print("\nThe oracle uses the ground-truth target. A model sees only the input, so the")
print("difference between these two rows is the part that is NOT learnable from the input.")
