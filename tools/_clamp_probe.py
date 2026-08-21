"""Is the clamp inside GlobalColorCorrection destroying the gradient?

refinement_head ends in Sigmoid, so its output is already in (0,1). GlobalColorCorrection then
applies image*gain + shift with a learned gain that trained to ~1.25, and clamps to [0,1].
Any pixel pushed outside the range gets zero gradient -- and with gain 1.25, shift -0.09,
everything above (1+0.09)/1.25 = 0.872 saturates.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import get_splits
from models.build import build_model
from utils.checkpoint import load_checkpoint

dev='cuda'
_, val = get_splits(augment_train=False)
loader = DataLoader(val, batch_size=8, shuffle=False, num_workers=0)
m = build_model(device=dev); load_checkpoint('checkpoints/best.pt', model=m, device=torch.device(dev)); m.eval()
gc = m.global_correction

clamped_hi=clamped_lo=tot=0
with torch.no_grad():
    for x,y in loader:
        x=x.to(dev)
        _,ctx=m.physics_encoder(x)
        saved=m.global_correction; m.global_correction=None
        base=m(x); m.global_correction=saved
        g,s = gc.predicted_params(ctx)
        raw = base*g.unsqueeze(-1).unsqueeze(-1) + s.unsqueeze(-1).unsqueeze(-1)
        clamped_hi += float((raw>1.0).sum()); clamped_lo += float((raw<0.0).sum()); tot += raw.numel()
print(f"pixels pushed ABOVE 1.0 (clamped, zero gradient) : {clamped_hi/tot*100:.2f}%")
print(f"pixels pushed BELOW 0.0 (clamped, zero gradient) : {clamped_lo/tot*100:.2f}%")
print(f"TOTAL pixels receiving NO gradient through the clamp: {(clamped_hi+clamped_lo)/tot*100:.2f}%")

# Re-run the direct fit WITHOUT the clamp and compare to the clamped result (18.4%).
cache=[]
with torch.no_grad():
    for x,y in loader:
        x,y=x.to(dev),y.to(dev)
        _,ctx=m.physics_encoder(x)
        saved=m.global_correction; m.global_correction=None
        base=m(x); m.global_correction=saved
        cache.append((base,ctx,y))

def fit(use_clamp, steps=300, lr=3e-3):
    import copy
    g2 = copy.deepcopy(gc)
    for p in g2.parameters(): p.requires_grad_(True)
    opt=torch.optim.Adam(g2.parameters(), lr=lr)
    def err():
        t=0.;n=0
        with torch.no_grad():
            for base,ctx,y in cache:
                gg,ss=g2.predicted_params(ctx)
                o=base*gg.unsqueeze(-1).unsqueeze(-1)+ss.unsqueeze(-1).unsqueeze(-1)
                if use_clamp: o=o.clamp(0,1)
                t+=float((o.mean(dim=(2,3))-y.mean(dim=(2,3))).abs().mean()); n+=1
        return t/n
    before=err()
    for _ in range(steps):
        for base,ctx,y in cache:
            opt.zero_grad()
            gg,ss=g2.predicted_params(ctx)
            o=base*gg.unsqueeze(-1).unsqueeze(-1)+ss.unsqueeze(-1).unsqueeze(-1)
            if use_clamp: o=o.clamp(0,1)
            F.l1_loss(o.mean(dim=(2,3)), y.mean(dim=(2,3))).backward()
            opt.step()
    after=err()
    # per-image variation and correlation with what is needed
    S=[];N=[]
    with torch.no_grad():
        for base,ctx,y in cache:
            gg,ss=g2.predicted_params(ctx); S.append(ss.cpu())
            N.append((y.mean(dim=(2,3))-base.mean(dim=(2,3))).cpu())
    S=torch.cat(S).numpy(); N=torch.cat(N).numpy()
    corr=[float(np.corrcoef(S[:,c],N[:,c])[0,1]) for c in range(3)]
    return before, after, corr

print()
for tag,use in [("WITH clamp (current code)", True), ("WITHOUT clamp", False)]:
    b,a,c = fit(use)
    print(f"{tag:<28} DC err {b:.5f} -> {a:.5f}  ({(1-a/b)*100:5.1f}% reduction)  corr={[round(v,3) for v in c]}")
