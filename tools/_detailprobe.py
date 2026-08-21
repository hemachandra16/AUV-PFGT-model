"""Does enhancement measurably destroy the fine detail that small classes rely on?

The ablation shows enhancement costs ~0.08-0.09 AP50 on small benthic classes (starfish,
scallop, echinus, holothurian) but only ~0.00-0.01 on large distinctive ones (jellyfish,
cuttlefish, turtle, diver). If that is texture loss rather than pure colour-domain shift,
enhanced frames should carry measurably less high-frequency energy.

Laplacian variance is the standard sharpness proxy; we also report high-pass energy.
"""
import sys, random
from pathlib import Path
import numpy as np, cv2

ROOT = Path(__file__).resolve().parents[1]
raw_dir = ROOT/"datasets/RUOD_yolo/images/val"
enh_dir = ROOT/"datasets/RUOD_yolo_enhanced/images/val"

names = sorted(p.name for p in enh_dir.glob("*.jpg"))
random.Random(0).shuffle(names)
names = names[:300]

def stats(p):
    im = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if im is None: return None
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float64)
    lap = cv2.Laplacian(g, cv2.CV_64F).var()
    blur = cv2.GaussianBlur(g,(0,0),2.0)
    hp = float(np.mean((g-blur)**2))
    return lap, hp, float(g.std())

rl=[];re_=[];rh=[];eh=[];rc=[];ec=[]
for n in names:
    a = stats(raw_dir/n); b = stats(enh_dir/n)
    if a and b:
        rl.append(a[0]); re_.append(b[0])
        rh.append(a[1]); eh.append(b[1])
        rc.append(a[2]); ec.append(b[2])

def m(x): return float(np.mean(x))
print(f"sampled {len(rl)} paired val images\n")
print(f"{'metric':<26}{'raw':>12}{'enhanced':>12}{'change':>12}")
print("-"*62)
print(f"{'Laplacian variance':<26}{m(rl):>12.1f}{m(re_):>12.1f}{(m(re_)/m(rl)-1)*100:>11.1f}%")
print(f"{'high-pass energy':<26}{m(rh):>12.2f}{m(eh):>12.2f}{(m(eh)/m(rh)-1)*100:>11.1f}%")
print(f"{'global contrast (std)':<26}{m(rc):>12.2f}{m(ec):>12.2f}{(m(ec)/m(rc)-1)*100:>11.1f}%")
share = sum(1 for a,b in zip(rl,re_) if b<a)/len(rl)*100
print(f"\nenhanced sharper?  lower Laplacian variance in {share:.0f}% of images")
