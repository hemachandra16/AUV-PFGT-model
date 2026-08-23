"""Extract figures from the two proof pages into a single JSON the report builders share.

Both `outputs/session4_proof.html` and `outputs/detection_proof.html` already contain
base64-embedded JPEGs that were rendered from held-out images. Re-rendering them here would
mean a second code path that could silently diverge from what those pages show, so this pulls
the exact bytes out of the published pages instead.

Enhancement rows carry five columns (original / first version / session-3 / session-4 /
reference). Only three are wanted for the report: the original, the *installed* model
(session 3, marked `class='hero'` in the proof page), and the human reference. Session 4's
column is deliberately dropped -- that model lost on the held-out set and is not what ships.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "_figures.json"

ROW_RE = re.compile(r"<div class='row'>(.*?)(?=<div class='row'>|<h2>|</body>)", re.S)
FIG_RE = re.compile(r"<figure class='([^']*)'><img src='(data:image/[^']+)'[^>]*>"
                    r"<figcaption>(.*?)</figcaption>", re.S)
DET_FIG_RE = re.compile(r"<figure><img src='(data:image/[^']+)'[^>]*>"
                        r"<figcaption>(.*?)</figcaption>", re.S)
VERDICT_RE = re.compile(r"<div class='verdict'><span class='tag[^']*'>([^<]*)</span>(.*?)</div>", re.S)


def clean(s: str) -> str:
    """Strip tags and unescape, so captions survive into a differently-styled page."""
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def enhancement_rows():
    txt = (ROOT / "outputs" / "session4_proof.html").read_text(encoding="utf-8")
    rows = []
    for block in ROW_RE.findall(txt):
        figs = FIG_RE.findall(block)
        if len(figs) != 5:
            continue
        vm = VERDICT_RE.search(block)
        cls, verdict = (clean(vm.group(1)), clean(vm.group(2))) if vm else ("", "")
        # column 2 carries class 'hero' in the proof page = the installed session-3 model
        installed = figs[2]
        rows.append({
            "kind": "enhancement",
            "verdict_tag": cls,
            "verdict": verdict,
            "panels": [
                {"src": figs[0][1], "label": "original photograph"},
                {"src": installed[1], "label": clean(installed[2]).replace("last session", "PFGT-UIE")},
                {"src": figs[4][1], "label": "human editor's reference"},
            ],
        })
    return rows


def detection_rows():
    txt = (ROOT / "outputs" / "detection_proof.html").read_text(encoding="utf-8")
    rows = []
    for block in ROW_RE.findall(txt):
        figs = DET_FIG_RE.findall(block)
        if len(figs) != 2:
            continue
        vm = VERDICT_RE.search(block)
        cls, verdict = (clean(vm.group(1)), clean(vm.group(2))) if vm else ("", "")
        rows.append({
            "kind": "detection",
            "verdict_tag": cls,
            "verdict": verdict,
            "panels": [
                {"src": figs[0][0], "label": clean(figs[0][1])},
                {"src": figs[1][0], "label": clean(figs[1][1])},
            ],
        })
    return rows


def main() -> None:
    enh, det = enhancement_rows(), detection_rows()
    print(f"enhancement rows parsed: {len(enh)}")
    print(f"detection rows parsed  : {len(det)}")
    for r in det:
        print(f"   [{r['verdict_tag']:<12}] {r['verdict'][:72]}")

    # Pick a spread rather than the first N: the report must show weak classes too.
    def pick(rows, tags):
        out, seen = [], set()
        for t in tags:
            for r in rows:
                if r["verdict_tag"].upper() == t and t not in seen:
                    out.append(r); seen.add(t); break
        return out

    det_pick = pick(det, ["CUTTLEFISH", "TURTLE", "CORALS", "SCALLOP"])
    if len(det_pick) < 3:                       # fall back rather than emit a thin gallery
        det_pick = det[:3]
    enh_pick = [enh[0], enh[len(enh) // 2], enh[-1]] if len(enh) >= 3 else enh

    data = {"enhancement": enh_pick, "detection": det_pick[:3]}
    OUT.write_text(json.dumps(data), encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"\nselected {len(enh_pick)} enhancement + {len(det_pick[:3])} detection rows")
    print(f"wrote {OUT} ({kb:.0f} KB)")
    for r in data["detection"]:
        print(f"   detection: {r['verdict_tag']}")
    for r in data["enhancement"]:
        print(f"   enhancement: {r['verdict_tag']} - {r['verdict'][:60]}")


if __name__ == "__main__":
    main()
