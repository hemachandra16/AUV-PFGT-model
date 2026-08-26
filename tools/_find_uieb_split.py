"""Third sweep, via the raw CDN instead of the rate-limited API.

Guesses the conventional locations a UIEB split list would live in, across every UIE repo that
can be harvested from the community list plus the known baselines. Any file that comes back
with roughly 90 image-looking lines is a candidate split, and is compared against the one
already found -- an identical list from an independent repo would be real evidence of a shared
standard; a *different* 90 would be evidence there is no standard at all, which is just as
useful an answer.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

PATHS = [
    "data/UIEB/test.txt", "data/test.txt", "test.txt", "datasets/UIEB/test.txt",
    "data/UIEB/test_list.txt", "data/test_list.txt", "test_list.txt", "splits/test.txt",
    "data/UIEB/val.txt", "UIEB/test.txt", "dataset/test.txt", "data/split/test.txt",
]
EXTRA_REPOS = [
    "ddz16/UIE_Benckmark", "LintaoPeng/U-shape-Transformer", "Li-Chongyi/Ucolor",
    "Huang-ShiRui/Semi-UIR", "JunlinHan/CWR", "piggy2009/DM_underwater",
    "Owen718/FiveAPlus-Network", "Li-Chongyi/Water-Net_Code",
    "pksvision/Deep-WaveNet-Underwater-Image-Restoration", "zhenqifu/PUIE-Net",
    "Fatemeh-Behrad/UIE", "Jzy2017/TACL", "wdhudiekou/UIE",
    "AiArt-HDU/HCLR-Net", "Kaikai-Zhao/UIE", "wangyanckxx/Single-Underwater-Image-Enhancement-and-Color-Restoration",
]
NAME_RE = re.compile(r"^[\w./\\-]+\.(png|jpg|jpeg|bmp)$", re.I)


def fetch(repo: str, br: str, path: str):
    url = f"https://raw.githubusercontent.com/{repo}/{br}/{path}"
    try:
        with urllib.request.urlopen(
                urllib.request.Request(url, headers={"User-Agent": "probe"}), timeout=20) as r:
            return url, r.read().decode("utf-8", "replace")
    except Exception:
        return url, None


def main() -> None:
    try:
        md = urllib.request.urlopen(urllib.request.Request(
            "https://raw.githubusercontent.com/lizhh268/"
            "awesome_underwater_image_enhancement-UIE-/main/README.md",
            headers={"User-Agent": "probe"}), timeout=30).read().decode("utf-8", "replace")
        harvested = {m for m in re.findall(r"github\.com/([\w.-]+/[\w.-]+)", md)
                     if not m.endswith((".png", ".jpg", ".git", ".svg"))}
    except Exception as e:
        print("awesome list unavailable:", e)
        harvested = set()

    repos = sorted(harvested | set(EXTRA_REPOS))
    jobs = [(r, b, p) for r in repos for b in ("main", "master") for p in PATHS]
    print(f"{len(repos)} repos x {len(PATHS)} paths x 2 branches = {len(jobs)} raw probes")

    hits = {}
    with cf.ThreadPoolExecutor(max_workers=16) as ex:
        for (repo, br, path), (url, body) in zip(
                jobs, ex.map(lambda j: fetch(*j), jobs)):
            if not body:
                continue
            lines = [l.strip() for l in body.splitlines() if l.strip()]
            if not lines or not all(NAME_RE.match(l) for l in lines[:20]):
                continue
            hits[url] = lines
            print(f"  LIST {url}  ({len(lines)} entries)")

    print(f"\n{len(hits)} filename lists retrieved")
    ref = Path("_t90_ddz16.txt")
    if ref.exists() and hits:
        known = set(ref.read_text(encoding="utf-8").split())
        print(f"\ncomparison against the ddz16 90-name list:")
        for url, lines in hits.items():
            s = set(lines)
            if len(s) < 40 or len(s) > 200:
                continue
            inter = len(s & known)
            print(f"  {url}\n     {len(s)} names, {inter} shared with ddz16 "
                  f"({100*inter/max(len(s),1):.0f}% overlap) -> "
                  f"{'IDENTICAL' if s == known else 'DIFFERENT'}")
    json.dump({k: v for k, v in hits.items()}, open("_split_lists.json", "w"), indent=1)


if __name__ == "__main__":
    main()
