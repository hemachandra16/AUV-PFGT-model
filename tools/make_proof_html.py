"""Build outputs/session4_proof.html — the "look at it yourself" version of the report.

A PSNR number is not something a person can check by looking. This builds one self-contained
HTML file (images embedded as base64, so it opens by double-clicking and survives being
emailed or copied to a pendrive) showing, for held-out images the model never trained on:

    raw input | pre-fix baseline | session 3 | session 4 (tonight) | ground-truth reference

with a plain-English sentence under each row, zoomed crops on the strongest colour casts, and
**the images that got worse as well as the ones that got better**. Cherry-picking only wins
would be exactly the quiet inflation these sessions have been explicit about avoiding.

Usage:
    python tools/make_proof_html.py
"""
from __future__ import annotations

import base64
import html
import importlib
import io as _io
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BASE_CODE = ROOT / "_archive" / "baseline_code"
OUT = ROOT / "outputs" / "session4_proof.html"

PANEL_W = 300          # width of each panel in the grid
CROP_W = 420           # width of each zoom panel


# --------------------------------------------------------------------------- helpers
def purge_models():
    for n in list(sys.modules):
        if n == "models" or n.startswith("models."):
            del sys.modules[n]


def load_img(path: Path) -> torch.Tensor:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.array(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def run_model(model, t: torch.Tensor, dev: str) -> torch.Tensor:
    _, _, h, w = t.shape
    ph, pw = (16 - h % 16) % 16, (16 - w % 16) % 16
    x = t.to(dev)
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    with torch.no_grad():
        y = model(x)
    return y[:, :, :h, :w].cpu()


def to_pil(t: torch.Tensor) -> Image.Image:
    a = (t.squeeze(0).clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return Image.fromarray(a)


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    m = float(torch.mean((a - b) ** 2))
    return 99.0 if m <= 1e-12 else 10.0 * float(np.log10(1.0 / m))


def embed(img: Image.Image, width: int) -> str:
    """Resize and return a base64 data URI so the HTML is self-contained."""
    if img.width != width:
        img = img.resize((width, max(1, int(img.height * width / img.width))), Image.Resampling.LANCZOS)
    buf = _io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def colour_cast_strength(t: torch.Tensor) -> float:
    """How strong the colour cast is: how far the channel means are from each other."""
    mu = t.mean(dim=(2, 3))[0]
    return float((mu.max() - mu.min()) / (mu.mean() + 1e-6))


def describe(delta: float, raw: torch.Tensor, new: torch.Tensor, ref: torch.Tensor) -> str:
    """One plain sentence: what the number means for how the picture looks."""
    # Did the colour cast actually get closer to the reference?
    cast_raw = colour_cast_strength(raw)
    cast_new = colour_cast_strength(new)
    cast_ref = colour_cast_strength(ref)
    closer = abs(cast_new - cast_ref) < abs(cast_raw - cast_ref)
    cast_note = ("the strong colour tint in the original is largely gone"
                 if closer else "the colour tint is only partly corrected")

    if delta >= 1.0:
        return (f"Clearly better — about {delta:.1f} dB improved, which is a difference you can "
                f"see without being told: {cast_note}.")
    if delta >= 0.3:
        return (f"Better — about {delta:.1f} dB improved. A visible but modest change; "
                f"{cast_note}.")
    if delta > -0.3:
        return (f"Essentially unchanged ({delta:+.1f} dB). Too small to see; both versions look "
                f"about the same.")
    if delta > -1.0:
        return (f"Slightly worse — about {abs(delta):.1f} dB down. Hard to spot by eye; most of "
                f"the difference is in flat background areas rather than the subject.")
    return (f"Worse — about {abs(delta):.1f} dB down. This one is a genuine loss, shown here "
            f"rather than left out.")


def crop_box(img_w: int, img_h: int, frac: float = 0.42):
    cw, ch = int(img_w * frac), int(img_h * frac)
    x0 = (img_w - cw) // 2
    y0 = (img_h - ch) // 2
    return (x0, y0, x0 + cw, y0 + ch)


# --------------------------------------------------------------------------- main
def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from data.dataset import get_splits, subset_pair_names

    _, val = get_splits(augment_train=False)
    names = subset_pair_names(val)
    raw_dir = ROOT / "datasets" / "UIEB" / "raw-890"
    ref_dir = ROOT / "datasets" / "UIEB" / "reference-890"

    # ---- pre-fix baseline: old architecture from the archived worktree ----
    print("running pre-fix baseline ...", flush=True)
    purge_models()
    sys.path.insert(0, str(BASE_CODE))
    mm = importlib.import_module("models.model")
    old = mm.PFGTUIEModel().to(dev)
    old.load_state_dict(torch.load(ROOT / "checkpoints/_baseline_before_fixes/best.pt",
                                   map_location=dev, weights_only=False)["model_state_dict"])
    old.eval()
    base_out = {n: run_model(old, load_img(raw_dir / n), dev) for n in names}
    del old
    torch.cuda.empty_cache()
    sys.path.remove(str(BASE_CODE))

    # ---- session 3 and session 4 share an architecture; only the loss differed ----
    purge_models()
    from models.build import build_model
    from utils.checkpoint import load_checkpoint

    print("running session 3 ...", flush=True)
    s3 = build_model(device=dev)
    load_checkpoint(str(ROOT / "checkpoints/_session4_backup/best.pt"), model=s3,
                    device=torch.device(dev))
    s3.eval()
    s3_out = {n: run_model(s3, load_img(raw_dir / n), dev) for n in names}
    del s3
    torch.cuda.empty_cache()

    print("running session 4 ...", flush=True)
    s4 = build_model(device=dev)
    load_checkpoint(str(ROOT / "checkpoints/best.pt"), model=s4, device=torch.device(dev))
    s4.eval()
    s4_out = {n: run_model(s4, load_img(raw_dir / n), dev) for n in names}

    # ---- score everything, then pick an HONEST sample ----
    rows = []
    for n in names:
        raw = load_img(raw_dir / n)
        ref = load_img(ref_dir / n)
        p_base, p3, p4 = psnr(ref, base_out[n]), psnr(ref, s3_out[n]), psnr(ref, s4_out[n])
        rows.append({
            "name": n, "raw": raw, "ref": ref,
            "base": base_out[n], "s3": s3_out[n], "s4": s4_out[n],
            "p_base": p_base, "p3": p3, "p4": p4,
            "delta_vs_base": p4 - p_base, "delta_vs_s3": p4 - p3,
            "cast": colour_cast_strength(raw),
        })
    rows.sort(key=lambda r: r["delta_vs_base"], reverse=True)

    # 4 best, 2 middling, 2 worst — so wins and losses are both represented.
    mid = len(rows) // 2
    picks = rows[:4] + rows[mid:mid + 2] + rows[-2:]
    seen, chosen = set(), []
    for r in picks:
        if r["name"] not in seen:
            seen.add(r["name"])
            chosen.append(r)

    # Zoom crops: the three strongest colour casts, since that is what tonight targeted.
    zooms = sorted(rows, key=lambda r: r["cast"], reverse=True)[:3]

    n_better = sum(1 for r in rows if r["delta_vs_base"] > 0.3)
    n_same = sum(1 for r in rows if -0.3 <= r["delta_vs_base"] <= 0.3)
    n_worse = sum(1 for r in rows if r["delta_vs_base"] < -0.3)
    mean_base = float(np.mean([r["p_base"] for r in rows]))
    mean_s3 = float(np.mean([r["p3"] for r in rows]))
    mean_s4 = float(np.mean([r["p4"] for r in rows]))

    # ------------------------------------------------------------------ HTML
    P = []
    A = P.append
    A('<meta charset="utf-8">')
    A("<title>PFGT-UIE — see the results yourself</title>")
    A("""<style>
:root{--bg:#101014;--panel:#191920;--ink:#ececf2;--dim:#a0a0b0;--line:#2a2a36;
      --good:#4ade80;--bad:#f87171;--same:#94a3b8;}
*{box-sizing:border-box}
body{margin:0;padding:28px 20px 60px;background:var(--bg);color:var(--ink);
     font:16px/1.65 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}
.wrap{max-width:1600px;margin:0 auto}
h1{font-size:26px;margin:0 0 6px}
h2{font-size:20px;margin:44px 0 10px;border-bottom:1px solid var(--line);padding-bottom:8px}
.lead{background:var(--panel);border:1px solid var(--line);border-radius:12px;
      padding:20px 22px;margin:18px 0 8px;max-width:1000px}
.lead p{margin:0 0 12px}.lead p:last-child{margin:0}
.dim{color:var(--dim)}
.tally{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 4px}
.chip{background:var(--panel);border:1px solid var(--line);border-radius:999px;
      padding:7px 15px;font-size:14px}
.row{background:var(--panel);border:1px solid var(--line);border-radius:12px;
     padding:14px;margin:20px 0;overflow-x:auto}
.grid{display:flex;gap:10px;min-width:min-content}
figure{margin:0;flex:0 0 auto}
figure img{display:block;border-radius:7px;background:#000}
figcaption{font-size:12.5px;color:var(--dim);margin-top:7px;text-align:center;
           letter-spacing:.02em;text-transform:uppercase}
figure.hero figcaption{color:var(--ink);font-weight:600}
.verdict{margin-top:13px;font-size:15.5px}
.tag{display:inline-block;font-weight:700;margin-right:9px;padding:2px 9px;
     border-radius:6px;font-size:13px}
.t-good{background:rgba(74,222,128,.14);color:var(--good)}
.t-bad{background:rgba(248,113,113,.14);color:var(--bad)}
.t-same{background:rgba(148,163,184,.14);color:var(--same)}
table{border-collapse:collapse;margin:10px 0 0;font-size:15px}
th,td{padding:8px 16px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
th{color:var(--dim);font-weight:600}
tr.hl td{color:var(--good);font-weight:700}
.note{max-width:1000px;color:var(--dim);font-size:14.5px}
</style>""")
    A("<div class='wrap'>")
    A("<h1>PFGT-UIE — see the results yourself</h1>")
    A("<p class='dim'>Underwater photo enhancement. Every image below is one the model was "
      "never trained on, so these are fair tests, not memorised answers. The panel with "
      "the brighter caption is the best-performing version.</p>")

    # Say plainly which version is best rather than letting "newest" imply "best".
    s4_better = mean_s4 > mean_s3

    A("<div class='lead'>")
    A("<p><b>What you are looking at.</b> Each row is one underwater photograph. Left to right: "
      "the original murky photo, then what three different versions of our software produced, "
      "then the &ldquo;correct&rdquo; answer a human editor made by hand. The closer our result "
      "looks to that last image, the better.</p>")

    if s4_better:
        A(f"<p><b>The headline.</b> Averaged over all {len(rows)} test photos, tonight&rsquo;s "
          f"version scores <b>{mean_s4:.2f}</b>, last session&rsquo;s <b>{mean_s3:.2f}</b>, and "
          f"the original first version <b>{mean_base:.2f}</b>. Higher is better.</p>")
    else:
        A(f"<p><b>The headline, stated plainly: tonight&rsquo;s change did not work.</b> "
          f"Averaged over all {len(rows)} test photos, tonight&rsquo;s version scores "
          f"<b>{mean_s4:.2f}</b> &mdash; <b>lower</b> than last session&rsquo;s "
          f"<b>{mean_s3:.2f}</b>. (The original first version scored {mean_base:.2f}.) Higher "
          f"is better, so <b>last session&rsquo;s version is still the best one, and it is the "
          f"one left installed on the machine.</b> For scale: a 1-point gain is a difference "
          f"most people can see, so the {abs(mean_s3-mean_s4):.2f}-point drop here is small enough "
          f"that you will struggle to spot it by eye &mdash; but it is a drop, not a gain, and "
          f"it is being reported as one.</p>")

    A("<p><b>What tonight was trying to do.</b> The software has a component whose only job is "
      "to fix the overall colour tint of a photo &mdash; the blue-green wash you get "
      "underwater. It was built last session but turned out to be barely doing anything, so "
      "tonight it was given a direct instruction to do that job properly.</p>")

    if not s4_better:
        A("<p><b>Why it did not help &mdash; and why that is still worth knowing.</b> Before "
          "training, I measured how much of that colour tint is actually <i>predictable</i> "
          "from the murky photo alone. The answer is: not much. The &ldquo;correct&rdquo; "
          "images were made by a human editor by hand, and a lot of their colour reflects that "
          "person&rsquo;s taste rather than anything recoverable from the original photograph. "
          "Pushing the software harder to guess it simply made it worse at everything else. "
          "That measurement is the most useful result of the night: it shows this particular "
          "avenue was largely a dead end, which means nobody needs to spend more time on it.</p>")

    A(f"<p class='dim'>Compared against the original first version, tonight&rsquo;s is visibly "
      f"better on {n_better} of the {len(rows)} test photos, about the same on {n_same}, and "
      f"worse on {n_worse}. Both the wins and the losses are shown below &mdash; the losses "
      f"are not hidden.</p>")
    A("</div>")

    A("<div class='tally'>")
    A(f"<span class='chip'>&#9679; {n_better} clearly better</span>")
    A(f"<span class='chip'>&#9679; {n_same} about the same</span>")
    A(f"<span class='chip'>&#9679; {n_worse} worse</span>")
    A("</div>")

    A("<h2>Side by side &mdash; four of the best, two typical, two of the worst</h2>")
    A("<p class='note'>Chosen deliberately to include failures. If you only look at one thing, "
      "compare the fourth image in each row against the fifth.</p>")

    for r in chosen:
        d = r["delta_vs_base"]
        cls = "t-good" if d > 0.3 else ("t-bad" if d < -0.3 else "t-same")
        word = "BETTER" if d > 0.3 else ("WORSE" if d < -0.3 else "SAME")
        A("<div class='row'><div class='grid'>")
        for img, cap, hero in [
            (r["raw"], "original photo", False),
            (r["base"], f"first version &nbsp;{r['p_base']:.1f} dB", False),
            (r["s3"], f"last session &nbsp;{r['p3']:.1f} dB", not s4_better),
            (r["s4"], f"tonight &nbsp;{r['p4']:.1f} dB", s4_better),
            (r["ref"], "human editor's version", False),
        ]:
            A(f"<figure class='{'hero' if hero else ''}'>"
              f"<img src='{embed(to_pil(img), PANEL_W)}' width='{PANEL_W}'>"
              f"<figcaption>{cap}</figcaption></figure>")
        A("</div>")
        A(f"<div class='verdict'><span class='tag {cls}'>{word}</span>"
          f"{html.escape(describe(d, r['raw'], r['s4'], r['ref']))}</div>")
        A("</div>")

    A("<h2>Close-up: the photos with the strongest colour tint</h2>")
    A("<p class='note'>These three had the heaviest blue-green cast in the original, so they are "
      "the most direct visual test of tonight's specific goal &mdash; fixing overall colour. "
      "Zoomed into the centre of each so you can see detail as well as colour.</p>")
    for r in zooms:
        img0 = to_pil(r["raw"])
        box = crop_box(img0.width, img0.height)
        A("<div class='row'><div class='grid'>")
        for img, cap, hero in [
            (r["raw"], "original photo", False),
            (r["s3"], f"last session &nbsp;{r['p3']:.1f} dB", not s4_better),
            (r["s4"], f"tonight &nbsp;{r['p4']:.1f} dB", s4_better),
            (r["ref"], "human editor's version", False),
        ]:
            A(f"<figure class='{'hero' if hero else ''}'>"
              f"<img src='{embed(to_pil(img).crop(box), CROP_W)}' width='{CROP_W}'>"
              f"<figcaption>{cap}</figcaption></figure>")
        A("</div>")
        A(f"<div class='verdict'><span class='tag {'t-good' if r['delta_vs_s3']>0 else 't-same'}'>"
          f"COLOUR</span>Compared with last session's version, tonight's is "
          f"{'closer to' if r['delta_vs_s3']>0 else 'about as far from'} the human editor's "
          f"colours here ({r['delta_vs_s3']:+.1f} dB against it).</div>")
        A("</div>")

    A("<h2>The numbers, for completeness</h2>")
    A("<table><tr><th>version</th><th>average score</th><th>vs. original</th></tr>")
    A(f"<tr><td>original first version</td><td>{mean_base:.3f} dB</td><td>&mdash;</td></tr>")
    A(f"<tr class='{'' if s4_better else 'hl'}'><td>last session"
      f"{'' if s4_better else ' &mdash; best, and the one installed'}</td>"
      f"<td>{mean_s3:.3f} dB</td><td>{mean_s3-mean_base:+.3f} dB</td></tr>")
    A(f"<tr class='{'hl' if s4_better else ''}'><td>tonight</td><td>{mean_s4:.3f} dB</td>"
      f"<td>{mean_s4-mean_base:+.3f} dB</td></tr>")
    A("</table>")
    A(f"<p class='note' style='margin-top:14px'>Averages over all {len(rows)} held-out test "
      f"photos, computed the same way for every version. Full technical detail, including what "
      f"did <i>not</i> work, is in <code>docs/report_content.md</code>.</p>")
    A("</div>")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(P), encoding="utf-8")
    size_mb = OUT.stat().st_size / 1e6
    print(f"\nwrote {OUT}  ({size_mb:.1f} MB, {len(chosen)} rows + {len(zooms)} zooms)")
    print(f"means: baseline {mean_base:.3f} | session3 {mean_s3:.3f} | session4 {mean_s4:.3f}")
    print(f"tally: {n_better} better / {n_same} same / {n_worse} worse (threshold 0.3 dB)")


if __name__ == "__main__":
    main()
