"""Extract LSUI.zip into the layout data/dataset.py expects.

LSUI ships as LSUI/input/N.jpg + LSUI/GT/N.jpg, already filename-paired 1:1. The only change
needed is an `lsui_` prefix, so a merged training pool can never collide with UIEB filenames
(UIEB has bare-numeric names like 5554.png alongside 100_img_.png).

Writes:  datasets/LSUI/input/lsui_N.jpg
         datasets/LSUI/GT/lsui_N.jpg
"""
from __future__ import annotations
import sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "_dsprobe" / "LSUI.zip"
DST = ROOT / "datasets" / "LSUI"

def main() -> None:
    if not SRC.exists():
        sys.exit(f"missing {SRC}")
    z = zipfile.ZipFile(SRC)
    (DST / "input").mkdir(parents=True, exist_ok=True)
    (DST / "GT").mkdir(parents=True, exist_ok=True)

    counts = {"input": 0, "GT": 0}
    for name in z.namelist():
        if name.endswith("/"):
            continue
        parts = name.split("/")
        if len(parts) < 3:
            continue
        folder, base = parts[-2], parts[-1]
        if folder not in ("input", "GT"):
            continue
        out = DST / folder / f"lsui_{base}"
        if not out.exists():
            out.write_bytes(z.read(name))
        counts[folder] += 1

    inp = {p.name for p in (DST / "input").iterdir()}
    gt = {p.name for p in (DST / "GT").iterdir()}
    print(f"  extracted: input={counts['input']}  GT={counts['GT']}")
    print(f"  on disk  : input={len(inp)}  GT={len(gt)}")
    print(f"  filename intersection (what dataset.py pairs on): {len(inp & gt)}")
    print(f"  unpaired : input-only={len(inp - gt)}  GT-only={len(gt - inp)}")
    if inp != gt:
        sys.exit("FAIL: input and GT filename sets differ -- pairing would silently drop images")
    print("  PASS: perfect 1:1 pairing")

if __name__ == "__main__":
    main()
