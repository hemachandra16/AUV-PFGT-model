"""Shared parse of `docs/report_content.md`, so the website and the PDF cannot drift apart.

The markdown is the single source of truth for wording and numbers. This module turns it into
a list of numbered sections with rendered HTML bodies; the two builders then wrap those bodies
in their own presentation. Anything that differs between the two outputs (an inline SVG on the
web versus a monospace diagram in print, a live hyperlink versus a printed filename) is handled
by the builder, never by keeping two copies of the prose.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "report_content.md"

SECTION_RE = re.compile(r"^## (\d+)\.\s+(.*)$", re.M)


@dataclass
class Section:
    number: int
    title: str
    md: str
    html: str = ""
    blocks: dict = field(default_factory=dict)   # fenced blocks lifted out before rendering


def _render(md_text: str) -> str:
    return markdown.markdown(md_text, extensions=["tables", "attr_list", "sane_lists"])


def _lift_fences(md_text: str, store: dict) -> str:
    """Pull fenced blocks out before markdown runs, leaving a placeholder.

    The ASCII pipeline diagram and the `math` display line need per-builder treatment (an
    inline SVG on the web, monospace in print), and markdown's own <pre><code> wrapper would
    make that harder to target. Placeholders keep the surrounding prose untouched.
    """
    def repl(m):
        lang, body = m.group(1) or "text", m.group(2)
        key = f"@@FENCE{len(store)}@@"
        store[key] = {"lang": lang, "body": body}
        return f"\n\n{key}\n\n"
    return re.sub(r"```(\w*)\n(.*?)```", repl, md_text, flags=re.S)


def load() -> tuple[dict, list[Section]]:
    """Return (front-matter fields, sections)."""
    text = SOURCE.read_text(encoding="utf-8")

    first = SECTION_RE.search(text)
    head, body = text[:first.start()], text[first.start():]

    # front matter: '# Title', '## Subtitle', then a bold standfirst paragraph
    lines = [l for l in head.split("\n") if l.strip() and not l.strip() == "---"]
    front = {
        "title": lines[0].lstrip("# ").strip(),
        "subtitle": lines[1].lstrip("# ").strip(),
        "standfirst": _render("\n".join(lines[2:])),
    }

    bounds = [(m.start(), int(m.group(1)), m.group(2)) for m in SECTION_RE.finditer(text)]
    sections = []
    for i, (start, num, title) in enumerate(bounds):
        end = bounds[i + 1][0] if i + 1 < len(bounds) else len(text)
        raw = text[start:end]
        raw = raw.split("\n", 1)[1]                      # drop the '## N. Title' line itself
        raw = re.sub(r"\n---\s*\n", "\n", raw).strip()   # section rules are presentation
        store: dict = {}
        sections.append(Section(num, title, raw, _render(_lift_fences(raw, store)), store))
    return front, sections


def restore_fences(html_text: str, blocks: dict, renderer) -> str:
    """Put fenced blocks back, formatted by the caller's `renderer(lang, body) -> html`."""
    for key, blk in blocks.items():
        # markdown wraps a bare placeholder line in <p>...</p>; swap the whole paragraph
        html_text = re.sub(rf"<p>\s*{re.escape(key)}\s*</p>", lambda m: renderer(blk["lang"], blk["body"]),
                           html_text)
        html_text = html_text.replace(key, renderer(blk["lang"], blk["body"]))
    return html_text


if __name__ == "__main__":
    front, secs = load()
    print(f"title    : {front['title']}")
    print(f"subtitle : {front['subtitle']}")
    print(f"sections : {len(secs)}")
    for s in secs:
        tables = s.html.count("<table>")
        print(f"  {s.number}. {s.title:<52} {len(s.html):>6} chars  "
              f"{tables} table(s)  {len(s.blocks)} fenced block(s)")
