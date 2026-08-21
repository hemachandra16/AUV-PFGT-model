"""raw | 115ep pre-fix baseline | tonight's converged post-fix | reference, on held-out images."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "_archive" / "baseline_code"

def purge():
    for n in list(sys.modules):
        if n == "models" or n.startswith("models."):
            del sys.modules[n]

def load(p):
    with Image.open(p) as im:
        im = im.convert("RGB"); a = np.array(im, dtype=np.float32)/255
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0)

def run(model, t, dev):
    _,_,h,w = t.shape
    ph,pw = (16-h%16)%16, (16-w%16)%16
    x = t.to(dev)
    if ph or pw: x = F.pad(x,(0,pw,0,ph),mode="reflect")
    with torch.no_grad(): y = model(x)
    return y[:,:,:h,:w].cpu()

def topil(z):
    return Image.fromarray((z.squeeze(0).clamp(0,1).numpy().transpose(1,2,0)*255).astype(np.uint8))

def lab(img, text, w=380):
    img = img.resize((w, int(img.height*w/img.width)))
    o = Image.new("RGB",(img.width,img.height+22),(18,18,22)); o.paste(img,(0,22))
    ImageDraw.Draw(o).text((5,6),text,fill=(235,235,240)); return o

def psnr(a,b):
    m = float(torch.mean((a-b)**2))
    return 99.0 if m<=1e-12 else 10*float(np.log10(1.0/m))

dev = "cuda"
from data.dataset import get_splits, subset_pair_names
_, val = get_splits(augment_train=False)
names = subset_pair_names(val)
picks = [names[i] for i in (0,12,25,40,60,80)]
raw_dir, ref_dir = ROOT/"datasets/UIEB/raw-890", ROOT/"datasets/UIEB/reference-890"

# --- OLD baseline (pre-fix architecture from the worktree) ---
purge(); sys.path.insert(0,str(BASE))
mm = importlib.import_module("models.model")
old = mm.PFGTUIEModel().to(dev)
old.load_state_dict(torch.load(ROOT/"checkpoints/_baseline_before_fixes/best.pt",
                               map_location=dev, weights_only=False)["model_state_dict"])
old.eval()
old_out = {n: run(old, load(raw_dir/n), dev) for n in picks}
del old; torch.cuda.empty_cache(); sys.path.remove(str(BASE))

# --- NEW converged model ---
purge()
from models.build import build_model
from utils.checkpoint import load_checkpoint
new = build_model(device=dev)
load_checkpoint("checkpoints/best.pt", model=new, device=torch.device(dev)); new.eval()

print(f"{'image':<16}{'PSNR ref|OLD':>14}{'PSNR ref|NEW':>14}{'PSNR raw|OLD':>14}{'PSNR raw|NEW':>14}")
print("-"*74)
so=sn=ro=rn=0.0
for n in picks:
    t = load(raw_dir/n); r = load(ref_dir/n)
    o = old_out[n]; w_ = run(new, t, dev)
    a,b,c,d = psnr(r,o), psnr(r,w_), psnr(t,o), psnr(t,w_)
    so+=a; sn+=b; ro+=c; rn+=d
    print(f"{n:<16}{a:>14.2f}{b:>14.2f}{c:>14.2f}{d:>14.2f}")
    panels=[lab(topil(t),"raw input (degraded)"),
            lab(topil(o),f"PRE-FIX baseline 115ep  PSNR={a:.1f}"),
            lab(topil(w_),f"POST-FIX converged 101ep  PSNR={b:.1f}"),
            lab(topil(r),"reference (ground truth)")]
    H=max(p.height for p in panels)
    g=Image.new("RGB",(sum(p.width for p in panels),H),(18,18,22)); x=0
    for p in panels: g.paste(p,(x,0)); x+=p.width
    g.save(ROOT/"outputs/_final_check_150ep"/f"cmp_{Path(n).stem}.png")
k=len(picks)
print("-"*74)
print(f"{'MEAN':<16}{so/k:>14.2f}{sn/k:>14.2f}{ro/k:>14.2f}{rn/k:>14.2f}")
print(f"\nsaved {k} panels -> outputs/_final_check_150ep/")
