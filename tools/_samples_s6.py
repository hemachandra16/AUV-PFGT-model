"""raw | session3 | stage1(union) | stage2(union+FT) | reference, on held-out images."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image, ImageDraw
from data.dataset import get_splits, subset_pair_names
from models.build import build_model
from utils.checkpoint import load_checkpoint
ROOT=Path(__file__).resolve().parents[1]

def load(p):
    with Image.open(p) as im:
        a=np.array(im.convert('RGB'),dtype=np.float32)/255
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0)
def run(m,t,dev='cuda'):
    _,_,h,w=t.shape; ph,pw=(16-h%16)%16,(16-w%16)%16
    x=t.to(dev)
    if ph or pw: x=F.pad(x,(0,pw,0,ph),mode='reflect')
    with torch.no_grad(): y=m(x)
    return y[:,:,:h,:w].cpu()
def topil(z): return Image.fromarray((z.squeeze(0).clamp(0,1).numpy().transpose(1,2,0)*255).astype(np.uint8))
def lab(img,t,w=300):
    img=img.resize((w,int(img.height*w/img.width)))
    o=Image.new("RGB",(img.width,img.height+22),(18,18,22)); o.paste(img,(0,22))
    ImageDraw.Draw(o).text((5,6),t,fill=(235,235,240)); return o
def psnr(a,b):
    m=float(torch.mean((a-b)**2)); return 99.0 if m<=1e-12 else 10*float(np.log10(1.0/m))

_,val=get_splits(augment_train=False); names=subset_pair_names(val)
picks=[names[i] for i in (0,12,25,40,60,80)]
raw_dir,ref_dir=ROOT/"datasets/UIEB/raw-890",ROOT/"datasets/UIEB/reference-890"

outs={}
for tag,ck in [("s3","checkpoints/_session6_backup/best.pt"),
               ("s1","checkpoints/_stage1_union.pt"),
               ("s2","checkpoints/_stage2_finetuned.pt")]:
    m=build_model(device='cuda'); load_checkpoint(ck,model=m,device=torch.device('cuda')); m.eval()
    outs[tag]={n:run(m,load(raw_dir/n)) for n in picks}
    del m; torch.cuda.empty_cache()

print(f"{'image':<16}{'session3':>10}{'stage1':>10}{'stage2':>10}")
print('-'*46); acc={'s3':0,'s1':0,'s2':0}
for n in picks:
    t=load(raw_dir/n); r=load(ref_dir/n)
    p={k:psnr(r,outs[k][n]) for k in outs}
    for k in acc: acc[k]+=p[k]
    print(f"{n:<16}{p['s3']:>10.2f}{p['s1']:>10.2f}{p['s2']:>10.2f}")
    panels=[lab(topil(t),"raw input"),
            lab(topil(outs['s3'][n]),f"session 3  {p['s3']:.1f}dB"),
            lab(topil(outs['s1'][n]),f"S6 stage1 union  {p['s1']:.1f}dB"),
            lab(topil(outs['s2'][n]),f"S6 stage2 +FT  {p['s2']:.1f}dB"),
            lab(topil(r),"reference")]
    H=max(x.height for x in panels)
    g=Image.new("RGB",(sum(x.width for x in panels),H),(18,18,22)); xo=0
    for x in panels: g.paste(x,(xo,0)); xo+=x.width
    g.save(ROOT/"outputs/_final_check_session6"/f"s6_{Path(n).stem}.png")
k=len(picks); print('-'*46)
print(f"{'MEAN':<16}{acc['s3']/k:>10.2f}{acc['s1']/k:>10.2f}{acc['s2']/k:>10.2f}")
print(f"\nsaved {k} panels -> outputs/_final_check_session6/")
