"""Download the RUOD underwater detection dataset from the Hugging Face Hub.

RUOD (Real-world Underwater Object Detection): 14,000 real underwater images across
10 marine classes (holothurian, echinus, scallop, starfish, fish, corals, diver,
cuttlefish, turtle, jellyfish). Public repo — no authentication required.

Uses direct parallel HTTP against the hub's resolve endpoint rather than
huggingface_hub.snapshot_download: the repo is ~14k small JPEGs, and snapshot_download
spent its time on per-file cache bookkeeping and rate-limit backoff (observed: it
stalled completely after ~700 files). Plain threaded GETs sustain ~7 files/s.

Resumable: files already present with non-zero size are skipped.

Usage:
    python tools/fetch_ruod.py                  # everything
    python tools/fetch_ruod.py --limit-train 4000 --limit-val 1200
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import ssl
import sys
import threading
import time
import urllib.request
from pathlib import Path

ssl._create_default_https_context = ssl._create_unverified_context

ROOT = Path(__file__).resolve().parents[1]
REPO = "Mortallll/RUOD"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/coco"
DEST = ROOT / "datasets" / "RUOD_raw" / "coco"

_lock = threading.Lock()
_done = 0
_failed: list[str] = []


def fetch_one(split: str, name: str, retries: int = 3) -> bool:
    global _done
    out = DEST / split / name
    if out.exists() and out.stat().st_size > 0:
        with _lock:
            _done += 1
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{split}/{name}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = resp.read()
            if not data:
                raise ValueError("empty response")
            tmp = out.with_suffix(out.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(out)
            with _lock:
                _done += 1
            return True
        except Exception:
            if attempt == retries - 1:
                with _lock:
                    _failed.append(f"{split}/{name}")
                return False
            time.sleep(1.5 * (attempt + 1))
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--limit-val", type=int, default=None)
    args = ap.parse_args()

    ann_dir = DEST / "annotations"
    tasks: list[tuple[str, str]] = []
    for split, ann_name, limit in [("train", "instances_train.json", args.limit_train),
                                   ("val", "instances_val.json", args.limit_val)]:
        ann_path = ann_dir / ann_name
        if not ann_path.exists():
            sys.exit(f"Missing {ann_path}; fetch the annotations first.")
        coco = json.load(open(ann_path, encoding="utf-8"))
        names = [Path(i["file_name"]).name for i in coco["images"]]
        names.sort()
        if limit is not None:
            names = names[:limit]
        tasks.extend((split, n) for n in names)

    total = len(tasks)
    print(f"RUOD: {total} images to ensure ({args.workers} workers)", flush=True)

    t0 = time.time()
    last_report = t0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(fetch_one, s, n) for s, n in tasks]
        for _ in cf.as_completed(futures):
            now = time.time()
            if now - last_report >= 30:
                with _lock:
                    d = _done
                rate = d / max(now - t0, 1e-9)
                eta = (total - d) / max(rate, 1e-9) / 60
                print(f"  {d}/{total}  {rate:.1f} files/s  ETA {eta:.1f} min  "
                      f"failed={len(_failed)}", flush=True)
                last_report = now

    dt = time.time() - t0
    print(f"\nDone: {_done}/{total} in {dt/60:.1f} min ({_done/max(dt,1e-9):.1f} files/s)")
    if _failed:
        print(f"FAILED {len(_failed)} files (first 10): {_failed[:10]}")
        (ROOT / "logs" / "ruod_failed.txt").write_text("\n".join(_failed), encoding="utf-8")
    else:
        print("No failures.")


if __name__ == "__main__":
    main()
