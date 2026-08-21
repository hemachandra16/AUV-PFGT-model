"""Does InstanceNorm strip the global colour cast the model is supposed to correct?

model.py normalises with InstanceNorm2d(affine=False) at several points, e.g.
  L89  ll_proj      = low_projection_norm(low_projection(ll))
  L125 physics_fused= physics_fusion_norm(physics_fusion_projection(physics_features))

InstanceNorm2d(affine=False) rescales EACH channel of EACH sample to zero mean and unit
variance over the spatial dims. A global colour cast is, almost by definition, a per-channel
shift and scale. So the concern: the normalisation may be deleting exactly the signal the
network exists to estimate.

Test: take a real image, apply a synthetic underwater-style cast (attenuate R, boost B), and
measure how much the tensor changes BEFORE vs AFTER each normalisation. If a representation
is cast-invariant, its features cannot tell the decoder what cast to remove.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, numpy as np
from PIL import Image
from models.build import build_model

dev='cuda'
m = build_model(device=dev).eval()

img = Image.open('datasets/UIEB/reference-890/708_img_.png').convert('RGB').resize((256,256))
x0 = torch.from_numpy(np.array(img,dtype=np.float32)/255).permute(2,0,1).unsqueeze(0).to(dev)

def cast(x, r, b):
    y = x.clone(); y[:,0]*=r; y[:,2]*=b
    return y.clamp(0,1)

variants = {'none':(1.0,1.0), 'mild':(0.75,1.15), 'strong':(0.45,1.35)}

def rel(a, b):
    return float((a-b).abs().mean() / (a.abs().mean()+1e-9)) * 100

print("relative change in each representation when a colour cast is applied")
print("(high % = the representation still carries the cast; ~0% = the cast has been erased)\n")
print(f"{'representation':<34}{'mild cast':>12}{'strong cast':>14}")
print('-'*60)

with torch.no_grad():
    base = {}
    for tag,(r,b) in variants.items():
        x = cast(x0, r, b)
        ll,lh,hl,hh = m.wavelet(x)
        pre  = m.low_projection(ll)
        post = m.low_projection_norm(pre)
        phys = m.physics_encoder(x)
        pphys_pre  = m.physics_fusion_projection(phys)
        pphys_post = m.physics_fusion_norm(pphys_pre)
        base[tag] = dict(input=x, ll=ll, ll_proj_pre=pre, ll_proj_post=post,
                         physics_raw=phys, phys_proj_pre=pphys_pre, phys_proj_post=pphys_post)

    rows = [('input image', 'input'),
            ('LL wavelet band', 'll'),
            ('low_projection  BEFORE norm', 'll_proj_pre'),
            ('low_projection  AFTER  norm', 'll_proj_post'),
            ('physics_encoder output (raw)', 'physics_raw'),
            ('physics_fusion  BEFORE norm', 'phys_proj_pre'),
            ('physics_fusion  AFTER  norm', 'phys_proj_post')]
    for label,key in rows:
        mild   = rel(base['mild'][key],   base['none'][key])
        strong = rel(base['strong'][key], base['none'][key])
        print(f"{label:<34}{mild:>11.2f}%{strong:>13.2f}%")

print("\nAlso: does the physics encoder see global statistics at all?")
print("  physics_encoder layers:", [type(mm).__name__ for mm in m.physics_encoder.modules()
                                    if type(mm).__name__ in ('Conv2d','AdaptiveAvgPool2d','Linear')])
print("  -> all Conv2d with 3x3 kernels, no global pooling: receptive field is local,")
print("     so per-channel image-wide means cannot be represented.")
