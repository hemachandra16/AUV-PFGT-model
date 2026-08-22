"""Check whether LSUI or EUVP contain any of UIEB's held-out 89 test images.

This is the single most important question in the dataset-expansion decision. Underwater
datasets are frequently assembled from overlapping public dive footage, and UIEB itself was
built from other people's material. If either candidate dataset contains a source image that
also appears in UIEB's held-out split, adding it to training would leak test data and
invalidate every PSNR number this project has reported across four sessions.

Exact checksums are useless here: a duplicate would almost certainly have been resized and
re-encoded (UIEB ships PNG at up to 1800px; LSUI ships JPEG; EUVP ships 256x256 JPEG). So this
uses perceptual hashing, which survives rescaling and recompression:

  * dHash (64-bit) — gradient direction between adjacent pixels; robust to scale/brightness.
  * aHash (64-bit) — mean threshold; a weaker second opinion, used to cross-check.

Any pair within the Hamming threshold is then re-checked by direct normalised correlation on
32x32 grayscale, and the closest hits are written out as side-by-side images for eyeballing —
because a hash collision is evidence, not proof.

Comparison is run against BOTH the held-out 89 (the leak-critical set) and the 801 training
images (duplication there is not a correctness problem, but is worth knowing about).
"""
from __future__ import annotations

import io
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "_dsprobe"

# A 64-bit dHash differing in <= 6 bits is a very strong near-duplicate signal; 10 is a loose
# net deliberately set wide so borderline cases surface for manual review rather than passing
# silently.
STRICT_BITS = 6
LOOSE_BITS = 10


def _gray(img: Image.Image, w: int, h: int) -> np.ndarray:
    return np.asarray(img.convert("L").resize((w, h), Image.Resampling.LANCZOS), dtype=np.float64)


def dhash(img: Image.Image) -> int:
    g = _gray(img, 9, 8)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def ahash(img: Image.Image) -> int:
    g = _gray(img, 8, 8)
    bits = (g > g.mean()).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def thumb(img: Image.Image) -> np.ndarray:
    """Normalised 32x32 grayscale, for correlation re-checks."""
    g = _gray(img, 32, 32)
    g = g - g.mean()
    n = np.linalg.norm(g)
    return (g / n).flatten() if n > 1e-9 else g.flatten()


def popcount(x: int) -> int:
    return bin(x).count("1")


def load_uieb():
    from data.dataset import get_splits, subset_pair_names
    train, val = get_splits(augment_train=False)
    held = set(subset_pair_names(val))
    train_names = set(subset_pair_names(train))
    raw_dir = ROOT / "datasets" / "UIEB" / "raw-890"
    out = []
    for name in sorted(held | train_names):
        p = raw_dir / name
        if not p.exists():
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            out.append({
                "name": name,
                "split": "HELD-OUT" if name in held else "train",
                "dh": dhash(im), "ah": ahash(im), "tb": thumb(im),
            })
    return out


def iter_lsui_dir():
    """Scan the EXTRACTED LSUI directory -- i.e. the exact files training would consume.

    Checking the zip proves the archive is clean; checking the laid-out directory proves the
    thing that will actually be fed to the model is clean. They should agree, but the second
    is the one that matters operationally, so it is what the training gate uses.
    """
    d = ROOT / "datasets" / "LSUI" / "input"
    if not d.exists():
        return
    for p in sorted(d.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            with Image.open(p) as im:
                yield f"LSUI/input/{p.name}", im.convert("RGB")
        except Exception:
            continue


def iter_lsui():
    zpath = PROBE / "LSUI.zip"
    if not zpath.exists():
        return
    z = zipfile.ZipFile(zpath)
    for n in z.namelist():
        if n.endswith("/") or "/input/" not in n:
            continue          # the raw side is what would match UIEB's raw side
        try:
            with z.open(n) as fh:
                im = Image.open(io.BytesIO(fh.read())).convert("RGB")
            yield n, im
        except Exception:
            continue


def iter_euvp():
    ppath = PROBE / "euvp_dark.parquet"
    if not ppath.exists():
        return
    import pyarrow.parquet as pq
    f = pq.ParquetFile(ppath)
    idx = 0
    for batch in f.iter_batches(batch_size=256, columns=["input_image"]):
        for v in batch.to_pydict()["input_image"]:
            b = v["bytes"] if isinstance(v, dict) else v
            try:
                im = Image.open(io.BytesIO(b)).convert("RGB")
                yield f"euvp_dark/{idx}", im
            except Exception:
                pass
            idx += 1


def scan(label, iterator, uieb, out_dir: Path):
    print("=" * 78)
    print(f"SCANNING {label} against {len(uieb)} UIEB images "
          f"({sum(1 for u in uieb if u['split']=='HELD-OUT')} held-out)")
    print("=" * 78)

    u_dh = np.array([u["dh"] for u in uieb], dtype=object)
    hits = []
    n = 0
    best_overall = 64
    for name, im in iterator:
        dh, ah = dhash(im), ahash(im)
        n += 1
        for i, u in enumerate(uieb):
            d = popcount(dh ^ u["dh"])
            if d < best_overall:
                best_overall = d
            if d <= LOOSE_BITS:
                a = popcount(ah ^ u["ah"])
                corr = float(np.dot(thumb(im), u["tb"]))
                hits.append({"cand": name, "uieb": u["name"], "split": u["split"],
                             "dhash_bits": d, "ahash_bits": a, "corr": corr,
                             "img": im.copy()})
        if n % 1000 == 0:
            print(f"  ...{n} scanned, {len(hits)} candidate hits so far", flush=True)

    print(f"  scanned {n} images")
    print(f"  closest dHash distance found anywhere : {best_overall} bits "
          f"(0 = identical, 64 = unrelated)")
    strict = [h for h in hits if h["dhash_bits"] <= STRICT_BITS]
    held_hits = [h for h in hits if h["split"] == "HELD-OUT"]
    held_strict = [h for h in strict if h["split"] == "HELD-OUT"]

    print(f"  candidate hits within {LOOSE_BITS} bits : {len(hits)}  "
          f"(of which held-out: {len(held_hits)})")
    print(f"  strong hits within {STRICT_BITS} bits   : {len(strict)}  "
          f"(of which held-out: {len(held_strict)})")

    if hits:
        hits.sort(key=lambda h: (h["dhash_bits"], -h["corr"]))
        print(f"\n  closest {min(10, len(hits))} candidate pairs:")
        print(f"    {'cand':<26}{'uieb':<18}{'split':<10}{'dbits':>6}{'abits':>6}{'corr':>8}")
        for h in hits[:10]:
            print(f"    {h['cand'][-26:]:<26}{h['uieb']:<18}{h['split']:<10}"
                  f"{h['dhash_bits']:>6}{h['ahash_bits']:>6}{h['corr']:>8.3f}")
        # write the closest few out for visual confirmation
        out_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = ROOT / "datasets" / "UIEB" / "raw-890"
        for k, h in enumerate(hits[:6]):
            try:
                with Image.open(raw_dir / h["uieb"]) as ui:
                    ui = ui.convert("RGB").resize((320, 240))
                ci = h["img"].resize((320, 240))
                canvas = Image.new("RGB", (640, 240))
                canvas.paste(ci, (0, 0)); canvas.paste(ui, (320, 0))
                canvas.save(out_dir / f"{label}_{k}_d{h['dhash_bits']}_c{h['corr']:.2f}.png")
            except Exception:
                pass
        print(f"\n  wrote closest pairs to {out_dir} (left=candidate, right=UIEB)")

    return {"scanned": n, "closest": best_overall, "loose": len(hits),
            "strict": len(strict), "held_loose": len(held_hits),
            "held_strict": len(held_strict)}


def main() -> None:
    print("loading UIEB reference hashes ...", flush=True)
    uieb = load_uieb()
    print(f"  {len(uieb)} UIEB images hashed\n")

    out_dir = PROBE / "leak_candidates"
    results = {}

    # Default: gate the EXTRACTED directory, which is what training consumes.
    # --zip additionally re-checks the source archive; --euvp adds the EUVP probe.
    if (ROOT / "datasets" / "LSUI" / "input").exists() and "--zip" not in sys.argv:
        results["LSUI(dir)"] = scan("LSUI(dir)", iter_lsui_dir(), uieb, out_dir)
    else:
        results["LSUI(zip)"] = scan("LSUI(zip)", iter_lsui(), uieb, out_dir)
    if "--euvp" in sys.argv:
        print()
        results["EUVP"] = scan("EUVP", iter_euvp(), uieb, out_dir)

    print()
    print("=" * 78)
    print("VERDICT")
    print("=" * 78)
    gate_ok = all(r["held_strict"] == 0 for r in results.values())
    for k, r in results.items():
        verdict = ("CLEAN" if r["held_strict"] == 0 else "LEAK RISK")
        print(f"  {k:<6} scanned {r['scanned']:>5}  closest {r['closest']:>2} bits  "
              f"held-out strong hits: {r['held_strict']}   -> {verdict}")
    print("\n  A strong hit against a HELD-OUT image would invalidate every PSNR number this")
    print("  project has reported. Any nonzero count must be inspected visually before use.")


if __name__ == "__main__":
    main()
