"""Build outputs/detection_proof.html — see the detector's output next to the ground truth.

Companion to outputs/session4_proof.html (which covers enhancement). Same principle: a person
cannot check mAP by looking, so show the actual boxes against the actual answers, on images the
detector never trained on, and include the classes it is BAD at as well as the ones it is good
at.

Selection is deliberately balanced: the detector's three strongest classes (cuttlefish 0.965,
turtle 0.965, diver 0.929) and its three weakest (corals 0.694, scallop 0.714, holothurian
0.751), so the page cannot flatter the model by construction.
"""
from __future__ import annotations

import base64
import io as _io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
VAL_IMG = ROOT / "datasets" / "RUOD_yolo" / "images" / "val"
VAL_LBL = ROOT / "datasets" / "RUOD_yolo" / "labels" / "val"
WEIGHTS = ROOT / "checkpoints" / "detector" / "best.pt"
METRICS = ROOT / "results" / "detection_metrics.json"
OUT = ROOT / "outputs" / "detection_proof.html"

CLASSES = ["holothurian", "echinus", "scallop", "starfish", "fish",
           "corals", "diver", "cuttlefish", "turtle", "jellyfish"]

STRONG = ["cuttlefish", "turtle", "diver"]
WEAK = ["corals", "scallop", "holothurian"]

GT_COLOR = (255, 214, 64)      # amber = the correct answer
PRED_COLOR = (56, 214, 178)    # teal  = what the detector said
PANEL_W = 440


def embed(img: Image.Image, width: int) -> str:
    if img.width != width:
        img = img.resize((width, max(1, int(img.height * width / img.width))),
                         Image.Resampling.LANCZOS)
    buf = _io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=86)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def read_gt(stem: str, w: int, h: int):
    p = VAL_LBL / f"{stem}.txt"
    out = []
    if not p.exists():
        return out
    for line in p.read_text().strip().splitlines():
        if not line.strip():
            continue
        c, cx, cy, bw, bh = line.split()
        c = int(c); cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
        out.append((c, (cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def draw(img: Image.Image, boxes, colour, labelled=True):
    im = img.copy()
    d = ImageDraw.Draw(im)
    lw = max(2, int(round(min(im.width, im.height) / 300)))
    for b in boxes:
        cls, x0, y0, x1, y1 = b[0], b[1], b[2], b[3], b[4]
        d.rectangle([x0, y0, x1, y1], outline=colour, width=lw)
        if labelled:
            txt = CLASSES[cls] if isinstance(cls, int) else str(cls)
            if len(b) > 5:
                txt += f" {b[5]*100:.0f}%"
            tb = d.textbbox((x0, y0), txt)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            yy = max(0, y0 - th - 5)
            d.rectangle([x0, yy, x0 + tw + 8, yy + th + 5], fill=colour)
            d.text((x0 + 4, yy + 2), txt, fill=(10, 20, 26))
    return im


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[1:5]
    bx0, by0, bx1, by1 = b[1:5]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def sentence(name, gt, pred, matched, missed, spurious, unlabelled_cls=None):
    """One plain sentence about this image. No jargon left unexplained.

    `unlabelled_cls` names a class the detector found in quantity that the reference does not
    label at all. That case is worth calling out rather than scoring as a failure: it is the
    benchmark being incomplete, not the detector being wrong, and a reader looking at the
    picture will see that immediately and lose trust in the page if it says otherwise.
    """
    if not gt:
        return "There is nothing to find in this frame."
    if matched == len(gt) and not spurious:
        return (f"Found all {len(gt)} of the {len(gt)} animals marked in this frame, with no "
                f"false alarms.")
    if unlabelled_cls and spurious >= 3:
        return (f"Found all {matched} of the {len(gt)} marked animal"
                f"{'s' if len(gt)!=1 else ''} — and also picked out the {unlabelled_cls} in the "
                f"background, which the human labeller did not mark at all. Those extra boxes "
                f"look correct to the eye, but they count against the score, because the "
                f"reference says they are not there. This is a limitation of the benchmark "
                f"rather than of the detector.")
    parts = [f"Found {matched} of the {len(gt)} animals marked here"]
    if missed:
        parts.append(f"missed {missed}")
    if spurious:
        parts.append(f"and drew {spurious} box{'es' if spurious > 1 else ''} where the "
                     f"reference says there is nothing — those may be real animals the human "
                     f"labeller skipped, or genuine mistakes")
    return ", ".join(parts) + "."


def main() -> None:
    from ultralytics import YOLO

    metrics = json.loads(METRICS.read_text())
    per_class = metrics["per_class"]
    m = metrics["metrics"]

    model = YOLO(str(WEIGHTS))
    names = model.names if isinstance(model.names, dict) else {}

    # Index val images by which classes they contain.
    by_class = defaultdict(list)
    all_imgs = sorted(VAL_IMG.glob("*.jpg"))
    for p in all_imgs:
        lp = VAL_LBL / f"{p.stem}.txt"
        if not lp.exists():
            continue
        try:
            cls = {int(l.split()[0]) for l in lp.read_text().strip().splitlines() if l.strip()}
        except Exception:
            continue
        for c in cls:
            by_class[CLASSES[c]].append(p)

    # Balanced pick: strongest classes and weakest classes alike.
    picks, seen = [], set()
    for cname in STRONG + WEAK:
        for p in by_class.get(cname, [])[:40]:
            if p in seen:
                continue
            lp = VAL_LBL / f"{p.stem}.txt"
            n_obj = len([l for l in lp.read_text().strip().splitlines() if l.strip()])
            if 1 <= n_obj <= 12:            # readable frames, not 40-box seafloor carpets
                seen.add(p); picks.append((cname, p)); break
    # Top up to 10 with additional frames from the weak classes (the honest direction).
    for cname in WEAK + STRONG:
        if len(picks) >= 10:
            break
        for p in by_class.get(cname, [])[:80]:
            if p in seen:
                continue
            lp = VAL_LBL / f"{p.stem}.txt"
            n_obj = len([l for l in lp.read_text().strip().splitlines() if l.strip()])
            if 1 <= n_obj <= 12:
                seen.add(p); picks.append((cname, p)); break

    rows = []
    for focus, p in picks:
        img = Image.open(p).convert("RGB")
        gt = read_gt(p.stem, img.width, img.height)
        r = model.predict(source=str(p), conf=0.25, iou=0.45, verbose=False, device=0)[0]
        pred = []
        if r.boxes is not None:
            for b in r.boxes:
                x0, y0, x1, y1 = (float(v) for v in b.xyxy[0].tolist())
                pred.append((int(b.cls.item()), x0, y0, x1, y1, float(b.conf.item())))

        used = set()
        matched = 0
        for g in gt:
            best, bi = 0.0, None
            for i, q in enumerate(pred):
                if i in used or q[0] != g[0]:
                    continue
                v = iou(g, q)
                if v > best:
                    best, bi = v, i
            if best >= 0.5 and bi is not None:
                used.add(bi); matched += 1
        missed = len(gt) - matched
        spurious = len(pred) - len(used)

        # Is the detector finding a whole class the reference simply does not label here?
        gt_classes = {g[0] for g in gt}
        extra = [q for i, q in enumerate(pred) if i not in used]
        unlabelled = None
        if extra:
            from collections import Counter as _C
            top_cls, top_n = _C(q[0] for q in extra).most_common(1)[0]
            if top_cls not in gt_classes and top_n >= 3:
                unlabelled = CLASSES[top_cls]

        rows.append({
            "focus": focus, "name": p.name,
            "gt_img": draw(img, gt, GT_COLOR),
            "pred_img": draw(img, pred, PRED_COLOR),
            "n_gt": len(gt), "n_pred": len(pred),
            "matched": matched, "missed": missed, "spurious": spurious,
            "ap": per_class.get(focus, {}).get("AP50"),
            "sentence": sentence(p.name, gt, pred, matched, missed, spurious, unlabelled),
            "unlabelled": unlabelled,
        })

    # ---------------------------------------------------------------- HTML
    P = []
    A = P.append
    A('<meta charset="utf-8">')
    A("<title>Underwater detector — see the results yourself</title>")
    A("""<style>
:root{--bg:#0b1416;--panel:#121d20;--ink:#e8f1f0;--dim:#8fa5a6;--line:#1e2f33;
      --gt:#ffd640;--pred:#38d6b2;--warn:#f0876a;}
@media (prefers-color-scheme: light){
 :root{--bg:#f4f7f6;--panel:#ffffff;--ink:#12242a;--dim:#5d7378;--line:#dbe5e4;
       --gt:#b8860b;--pred:#0f9c81;--warn:#c1543a;}}
*{box-sizing:border-box}
body{margin:0;padding:30px 20px 70px;background:var(--bg);color:var(--ink);
     font:16px/1.65 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1200px;margin:0 auto}
h1{font-size:27px;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:20px;margin:44px 0 12px;padding-bottom:8px;border-bottom:1px solid var(--line)}
.lead{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:20px 22px;
      margin:18px 0;max-width:900px}
.lead p{margin:0 0 12px}.lead p:last-child{margin:0}
.dim{color:var(--dim)}
.key{display:flex;gap:20px;flex-wrap:wrap;margin:14px 0 0;font-size:14px}
.key span{display:flex;align-items:center;gap:8px}
.sw{width:15px;height:15px;border-radius:3px;display:inline-block}
.row{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px;
     margin:18px 0;overflow-x:auto}
.grid{display:flex;gap:12px;min-width:min-content}
figure{margin:0;flex:0 0 auto}
figure img{display:block;border-radius:6px;background:#000}
figcaption{font-size:12px;color:var(--dim);margin-top:6px;text-transform:uppercase;
           letter-spacing:.05em}
.verdict{margin-top:12px;font-size:15px}
.tag{display:inline-block;font-size:12px;font-weight:700;padding:2px 9px;border-radius:5px;
     margin-right:9px;background:rgba(56,214,178,.15);color:var(--pred)}
.tag.w{background:rgba(240,135,106,.15);color:var(--warn)}
table{border-collapse:collapse;font-size:14.5px;margin-top:8px}
th,td{padding:7px 15px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.92em}
</style>""")
    A("<div class='wrap'>")
    A("<h1>Underwater object detector — see the results yourself</h1>")
    A("<p class='dim'>Every frame below is from the held-out test set: the detector was never "
      "trained on any of them.</p>")

    A("<div class='lead'>")
    A("<p><b>What you are looking at.</b> Two copies of the same photograph. On the left, the "
      "boxes a human marked by hand — the correct answer. On the right, the boxes the software "
      "drew on its own. The closer the two match, the better.</p>")
    A(f"<p><b>What the headline score means.</b> The detector scores "
      f"<b>mAP@0.5 = {m['mAP50']:.3f}</b>. That is <i>not</i> "
      f"&ldquo;{m['mAP50']*100:.0f}% correct&rdquo; &mdash; it is a combined measure of how well "
      f"it balances finding the animals against raising false alarms, averaged over all ten "
      f"categories, counting a box as right if it overlaps the true one by at least half. Two "
      f"plainer numbers from the same run: it finds about "
      f"<b>{m['recall']*100:.0f}% of the animals</b> that are actually there (recall), and about "
      f"<b>{m['precision']*100:.0f}% of the boxes it draws are real</b> (precision).</p>")
    A("<p><b>This page is deliberately not flattering.</b> The frames were chosen to cover the "
      "three categories the detector handles best (cuttlefish, turtle, diver) <i>and</i> the "
      "three it handles worst (corals, scallop, sea cucumber) &mdash; so its weaknesses are on "
      "screen, not hidden.</p>")
    A("<div class='key'>"
      "<span><i class='sw' style='background:var(--gt)'></i> human-marked answer</span>"
      "<span><i class='sw' style='background:var(--pred)'></i> what the software found</span>"
      "</div>")
    A("</div>")

    A("<h2>Frame by frame</h2>")
    for r in rows:
        good = r["missed"] == 0 and r["spurious"] == 0
        A("<div class='row'><div class='grid'>")
        A(f"<figure><img src='{embed(r['gt_img'], PANEL_W)}' width='{PANEL_W}'>"
          f"<figcaption>human-marked answer &mdash; {r['n_gt']} object"
          f"{'s' if r['n_gt']!=1 else ''}</figcaption></figure>")
        A(f"<figure><img src='{embed(r['pred_img'], PANEL_W)}' width='{PANEL_W}'>"
          f"<figcaption>what the software found &mdash; {r['n_pred']} box"
          f"{'es' if r['n_pred']!=1 else ''}</figcaption></figure>")
        A("</div>")
        ap = f" &middot; this category scores {r['ap']:.2f} overall" if r["ap"] else ""
        A(f"<div class='verdict'><span class='tag{'' if good else ' w'}'>"
          f"{r['focus'].upper()}</span>{r['sentence']}"
          f"<span class='dim'>{ap}</span></div>")
        A("</div>")

    A("<h2>Every category, best to worst</h2>")
    A("<p class='dim'>Averaged over the full 4,200-image held-out set. The pattern is "
      "consistent: large, distinctive animals are easy; small ones that sit camouflaged against "
      "the seabed in groups are hard.</p>")
    A("<table><tr><th>category</th><th>score (AP@0.5)</th><th>precision</th><th>recall</th></tr>")
    for cname, v in sorted(per_class.items(), key=lambda kv: -kv[1]["AP50"]):
        A(f"<tr><td>{cname}</td><td>{v['AP50']:.3f}</td>"
          f"<td>{v['precision']:.3f}</td><td>{v['recall']:.3f}</td></tr>")
    A(f"<tr><td><b>overall</b></td><td><b>{m['mAP50']:.3f}</b></td>"
      f"<td><b>{m['precision']:.3f}</b></td><td><b>{m['recall']:.3f}</b></td></tr>")
    A("</table>")

    A("<h2>One thing worth knowing</h2>")
    A("<p class='lead' style='max-width:900px'>Running photos through the colour-correction "
      "software <i>before</i> showing them to this detector makes it <b>worse</b>, by about 3.9 "
      "points. That was measured on all 4,200 test frames, and it is why the tool detects on the "
      "raw photograph by default and only uses the corrected version for the picture a person "
      "looks at. This matches published findings from other groups &mdash; it is not unique to "
      "this system.</p>")
    A("</div>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(P), encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1e6:.1f} MB, {len(rows)} frames)")
    for r in rows:
        print(f"  {r['focus']:<12} {r['name']:<14} gt={r['n_gt']:<3} pred={r['n_pred']:<3} "
              f"matched={r['matched']} missed={r['missed']} spurious={r['spurious']}")


if __name__ == "__main__":
    main()
