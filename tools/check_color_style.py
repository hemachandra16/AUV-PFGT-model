"""Did the two-stage recipe actually pull the model's colour back to UIEB's convention?

The feasibility report measured that the datasets disagree on what "corrected" means:
    UIEB reference  R/B = 0.807     LSUI GT  R/B = 0.967
Stage 1 trains on the union, so its output should drift warm toward LSUI. Stage 2 fine-tunes on
UIEB alone, so its output should come back. This measures whether that actually happened --
independently of whether PSNR moved, because the mitigation can work mechanically and still not
pay off in the metric.
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from data.dataset import get_splits, subset_pair_names
from models.build import build_model
from utils.checkpoint import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]

def style(arr_list):
    """Mean per-channel stats over a list of (3,H,W) tensors in [0,1]."""
    R=G=B=cast=sat=0.0; n=0
    for a in arr_list:
        mu = a.mean(dim=(1,2))
        R+=float(mu[0]); G+=float(mu[1]); B+=float(mu[2])
        cast += float((mu.max()-mu.min())/mu.mean())
        sat += float((a.max(0).values - a.min(0).values).mean())
        n+=1
    return dict(R=R/n, G=G/n, B=B/n, RG=(R/n)/(G/n), RB=(R/n)/(B/n), cast=cast/n, sat=sat/n)

dev='cuda'
_, val = get_splits(augment_train=False)
names = subset_pair_names(val)
raw_dir = ROOT/"datasets/UIEB/raw-890"; ref_dir = ROOT/"datasets/UIEB/reference-890"

def load(p):
    with Image.open(p) as im:
        a=np.array(im.convert('RGB').resize((256,256)),dtype=np.float32)/255
    return torch.from_numpy(a).permute(2,0,1)

raws=[load(raw_dir/n) for n in names]
refs=[load(ref_dir/n) for n in names]

def run(ckpt):
    m=build_model(device=dev); load_checkpoint(ckpt, model=m, device=torch.device(dev)); m.eval()
    outs=[]
    with torch.no_grad():
        for i in range(0,len(raws),8):
            b=torch.stack(raws[i:i+8]).to(dev)
            outs.extend(list(m(b).cpu()))
    del m; torch.cuda.empty_cache()
    return outs

rows=[("UIEB raw (input)", style(raws)),
      ("UIEB reference (the TARGET)", style(refs))]
for label, ck in [("session 3 (UIEB only)", "checkpoints/_session6_backup/best.pt"),
                  ("stage 1 (union, no FT)", "checkpoints/_stage1_union.pt"),
                  ("stage 2 (union + UIEB FT)", "checkpoints/_stage2_finetuned.pt")]:
    rows.append((label, style(run(ck))))

# LSUI's own target style, for the other end of the scale
lsui=[load(p) for p in sorted((ROOT/"datasets/LSUI/GT").iterdir())[:300]]
rows.append(("LSUI GT (the OTHER convention)", style(lsui)))

print(f"{'':<32}{'R':>8}{'G':>8}{'B':>8}{'R/G':>8}{'R/B':>8}{'cast':>8}{'sat':>8}")
print("-"*88)
for lab,s in rows:
    print(f"{lab:<32}{s['R']:>8.3f}{s['G']:>8.3f}{s['B']:>8.3f}{s['RG']:>8.3f}{s['RB']:>8.3f}{s['cast']:>8.3f}{s['sat']:>8.3f}")

tgt = dict(rows)["UIEB reference (the TARGET)"]['RB']
print(f"\nDistance from UIEB's target R/B ({tgt:.3f}) -- lower is better calibrated:")
for lab in ["session 3 (UIEB only)","stage 1 (union, no FT)","stage 2 (union + UIEB FT)"]:
    v=dict(rows)[lab]['RB']
    print(f"   {lab:<32} R/B {v:.3f}   |diff| {abs(v-tgt):.3f}")
s1=abs(dict(rows)["stage 1 (union, no FT)"]['RB']-tgt); s2=abs(dict(rows)["stage 2 (union + UIEB FT)"]['RB']-tgt)
print(f"\n   stage 2 moved the calibration {'CLOSER TO' if s2<s1 else 'FURTHER FROM'} UIEB's convention "
      f"({s1:.3f} -> {s2:.3f}, {(1-s2/s1)*100:+.0f}% change in error)")
