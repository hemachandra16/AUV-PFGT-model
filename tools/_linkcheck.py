"""Resolve every relative markdown link in every tracked file against the working tree."""
import pathlib
import re
import subprocess
import sys
import urllib.parse

ROOT = pathlib.Path("D:/PhysicsFreqTransformer")
LINK = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

tracked = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split()
broken, checked, external = [], 0, 0
for f in tracked:
    if not f.endswith((".md", ".html")):
        continue
    try:
        text = (ROOT / f).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        continue
    for label, target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
            external += 1
            continue
        path_part = urllib.parse.unquote(target.split("#", 1)[0])
        if not path_part:
            continue
        checked += 1
        if not ((ROOT / f).parent / path_part).resolve().exists():
            broken.append((f, label[:40], target))

print(f"relative links resolved : {checked}")
print(f"external links          : {external}")
print(f"BROKEN                  : {len(broken)}")
for f, label, target in broken:
    print(f"   {f}  [{label}] -> {target}")
sys.exit(1 if broken else 0)
