"""Render `docs/report_content.md` into outputs/PFGT-UIE_report.pdf.

Same source as the website, different renderer, so the two cannot disagree on a number.

xhtml2pdf is not a browser. It has no CSS custom properties, no grid, no flexbox and no
`@media`, so the print stylesheet is written from scratch rather than reused -- everything is
block flow and tables, with literal colours. Three things needed specific handling:

* **The pipeline diagram.** xhtml2pdf renders SVG through svglib, but only from a file and only
  from presentation attributes -- the web diagram's CSS classes and `var()` colours resolve to
  nothing. So the web SVG is mechanically rewritten into a self-styled one. Deriving it from
  `build_website.PIPELINE_SVG` rather than retyping the coordinates means the two diagrams
  cannot drift apart, and an assertion below fails the build if the rewrite stops matching.
* **Fonts.** The webfonts are not embedded; print uses the PDF base-14 (Helvetica for labelling,
  Times for body). No font files to ship, nothing to fail to load.
* **Links to the proof pages** become filenames in print, because a hyperlink to a local HTML
  file is useless to somebody holding only the PDF.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from xhtml2pdf import pisa

import build_website as bw          # reuse the tested table-alignment pass and the diagram
import report_common as rc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "PFGT-UIE_report.pdf"
BUILD = ROOT / "outputs" / "_pdfbuild"
FIGS = ROOT / "docs" / "_figures.json"

INK, INK2, RULE, SIGNAL, AMBER, PAPER = "#12201f", "#4c6065", "#c2ced0", "#0b6d5c", "#8a6108", "#f2f5f6"


def print_svg() -> Path:
    """Rewrite the website's pipeline SVG into one svglib can style, and write it to disk."""
    s = re.search(r"<svg.*?</svg>", bw.PIPELINE_SVG, re.S).group(0)
    s = s.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="470" ', 1)
    subs = [
        ('fill="var(--signal)"', f'fill="{SIGNAL}"'),
        ('fill="var(--signal-2)"', f'fill="{AMBER}"'),
        ('<g class="dg-flow">',
         f'<g fill="none" stroke="{SIGNAL}" stroke-width="1.4" marker-end="url(#ah)">'),
        ('<g class="dg-prior">',
         f'<g fill="none" stroke="{AMBER}" stroke-width="1.2" stroke-dasharray="4 3" '
         f'marker-end="url(#ap)">'),
        ('<g class="dg-box dg-prior-box">',
         f'<g fill="none" stroke="{AMBER}" stroke-width="1.1" stroke-dasharray="5 3" '
         f'text-anchor="middle">'),
        ('<g class="dg-box">',
         f'<g fill="#ffffff" stroke="{INK}" stroke-width="1.2" text-anchor="middle">'),
        ('<g class="dg-legend">', '<g fill="none" text-anchor="middle">'),
        # text must reset stroke, or it inherits the box outline and prints smeared
        ('<text class="t1"',
         f'<text stroke="none" fill="{INK}" font-family="Helvetica" font-weight="bold" font-size="10.5"'),
        ('<text class="t2"',
         f'<text stroke="none" fill="{INK2}" font-family="Helvetica" font-size="8"'),
        ('<path class="dg-flow"', f'<path fill="none" stroke="{SIGNAL}" stroke-width="1.4"'),
        ('<path class="dg-prior"',
         f'<path fill="none" stroke="{AMBER}" stroke-width="1.2" stroke-dasharray="4 3"'),
    ]
    for a, b in subs:
        s = s.replace(a, b)
    assert "class=" not in s, f"unconverted class attribute left in print SVG: {s[s.find('class='):][:80]}"
    assert "var(--" not in s, "unconverted CSS variable left in print SVG"
    BUILD.mkdir(parents=True, exist_ok=True)
    p = BUILD / "pipeline.svg"
    p.write_text(s, encoding="utf-8")
    return p


CSS = f"""
@page {{
  size: a4 portrait;
  margin: 2.1cm 1.9cm 2.3cm 1.9cm;
  @frame footer {{ -pdf-frame-content: footerContent; bottom: 1.1cm;
                   margin-left: 1.9cm; margin-right: 1.9cm; height: 1cm; }}
}}
body {{ font-family: Times; font-size: 10.2pt; line-height: 145%; color: {INK}; }}
p {{ margin: 0 0 7pt 0; text-align: justify; }}
h1 {{ font-family: Helvetica; font-size: 27pt; margin: 0 0 4pt 0; color: {INK};
     -pdf-outline: true; -pdf-outline-level: 0; }}
h2 {{ font-family: Helvetica; font-size: 15.5pt; margin: 0 0 9pt 0; color: {INK};
     -pdf-outline: true; -pdf-outline-level: 1; -pdf-keep-with-next: true; }}
h3 {{ font-family: Helvetica; font-size: 11pt; margin: 13pt 0 4pt 0; color: {INK};
     -pdf-outline: true; -pdf-outline-level: 2; -pdf-keep-with-next: true; }}
h4 {{ font-family: Helvetica; font-size: 9pt; margin: 10pt 0 3pt 0; color: {INK2}; }}
ul, ol {{ margin: 0 0 7pt 12pt; }}
li {{ margin-bottom: 3pt; text-align: justify; }}
code {{ font-family: Courier; font-size: 8.6pt; }}
a {{ color: {SIGNAL}; }}
strong, b {{ font-family: Times; font-weight: bold; }}

.plate {{ font-family: Helvetica; font-size: 7.5pt; color: {SIGNAL}; letter-spacing: 1.4pt;
         margin: 0 0 3pt 0; }}
.sec {{ -pdf-keep-with-next: true; }}
blockquote {{ margin: 8pt 0; padding: 6pt 9pt; background: {PAPER};
             border-left: 2.4pt solid {AMBER}; }}
blockquote p {{ margin: 0; }}

table {{ width: 100%; margin: 9pt 0; -pdf-keep-with-next: false; }}
th {{ font-family: Helvetica; font-size: 7.6pt; color: {INK2}; text-align: left;
     padding: 4pt 5pt; border-bottom: 0.9pt solid {INK}; border-top: 0.9pt solid {INK}; }}
td {{ font-family: Helvetica; font-size: 8.2pt; padding: 3.6pt 5pt;
     border-bottom: 0.4pt solid {RULE}; }}
.ar {{ text-align: right; }}

pre {{ font-family: Courier; font-size: 7.4pt; background: {PAPER};
      border-left: 2pt solid {RULE}; padding: 6pt 8pt; margin: 8pt 0; }}
.formula {{ font-family: Courier; font-size: 11pt; color: {SIGNAL}; text-align: center;
           padding: 9pt 0; margin: 8pt 0; border-top: 0.6pt solid {RULE};
           border-bottom: 0.6pt solid {RULE}; }}
.caption {{ font-family: Helvetica; font-size: 7.6pt; color: {INK2}; margin: 3pt 0 10pt 0; }}
.panelcap {{ font-family: Helvetica; font-size: 6.6pt; color: {INK2}; padding: 2pt 3pt 0 3pt;
            text-align: center; }}
.gcell {{ padding: 0 2pt; text-align: center; }}
.verdict {{ font-family: Times; font-size: 8.8pt; color: {INK2}; padding: 4pt 3pt 9pt 3pt; }}
.tag {{ font-family: Helvetica; font-size: 7pt; color: {SIGNAL}; }}
.tagw {{ font-family: Helvetica; font-size: 7pt; color: {AMBER}; }}

#footerContent {{ font-family: Helvetica; font-size: 7.4pt; color: {INK2};
                 border-top: 0.4pt solid {RULE}; padding-top: 3pt; }}

.cover {{ margin-top: 120pt; }}
.cover h1 {{ font-size: 40pt; line-height: 100%; }}
.cover .sub {{ font-family: Helvetica; font-size: 13pt; color: {INK}; margin: 10pt 0 0 0;
              line-height: 130%; }}
.cover .stand {{ font-size: 10.5pt; color: {INK2}; margin: 16pt 0 0 0; }}
.cover .eyebrow {{ font-family: Helvetica; font-size: 8pt; color: {SIGNAL}; letter-spacing: 2pt; }}
.spec td {{ border-bottom: 0.4pt solid {RULE}; border-top: 0.9pt solid {INK}; padding: 7pt 0; }}
.spec .k {{ font-family: Helvetica; font-size: 7.6pt; color: {INK2}; }}
.spec .v {{ font-family: Helvetica; font-size: 13pt; color: {SIGNAL}; }}
.spec .vw {{ font-family: Helvetica; font-size: 11pt; color: {AMBER}; }}
.toc td {{ border: none; padding: 4pt 0; font-family: Helvetica; font-size: 9.6pt; }}
.toc .n {{ color: {SIGNAL}; font-size: 8pt; }}
.newpage {{ page-break-before: always; }}
"""


# reportlab's base-14 fonts encode WinAnsi only. Most of what this report uses survives
# (arrows, lambda, radical, superscript two), but a few characters render as a .notdef box --
# caught by looking at the rendered pages, not by the builder, which reports no error at all.
# Substitute those for print; the markdown and the website keep the real glyphs.
PRINT_GLYPHS = {
    "ᵀ": "^T",      # MODIFIER LETTER CAPITAL T -- printed as a black box in Courier
    "⁻": "^-",      # SUPERSCRIPT MINUS
    "⁷": "7",       # SUPERSCRIPT SEVEN
    "←": "&lt;-",   # LEFTWARDS ARROW
}


def substitute_glyphs(html: str) -> str:
    for a, b in PRINT_GLYPHS.items():
        html = html.replace(a, b)
    return html


def fence_renderer_factory(svg_path: Path):
    def render(lang: str, body: str) -> str:
        if lang == "math":
            return f'<div class="formula">{body.strip()}</div>'
        if "Haar DWT" in body and "─" in body:
            return (f'<div><img src="{svg_path.resolve().as_posix()}" width="650"></div>'
                    f'<div class="caption">Figure 1 &mdash; the enhancement pipeline. There is '
                    f'deliberately no global input&rarr;output skip connection: the network must '
                    f'reconstruct the image rather than learn a residual on it.</div>')
        esc = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f"<pre>{esc.rstrip()}</pre>"
    return render


def gallery(rows, width: int, title: str, start_fig: int) -> str:
    """Lay a figure gallery out on its own page.

    xhtml2pdf ignores `page-break-inside: avoid` and will split a table between its image
    row and its caption row, stranding a row of photographs from the sentence that explains
    it -- which is exactly what it did before this page break was made explicit. Each
    gallery is sized to fit one page whole, so there is nothing left to split.
    """
    out = [f'<div class="newpage"><h4>{title}</h4>']
    for n, r in enumerate(rows, start=start_fig):
        cells, caps = [], []
        for p in r["panels"]:
            cells.append(f'<td class="gcell"><img src="{p["src"]}" width="{width}"></td>')
            caps.append(f'<td class="panelcap">{p["label"]}</td>')
        tag_cls = "tagw" if r["verdict_tag"].upper() in (
            "WORSE", "TURTLE", "CORALS", "SCALLOP", "HOLOTHURIAN") else "tag"
        # keep the panels, their captions and the verdict on one page -- split across a
        # break, a row of photographs is stranded from the sentence explaining it
        out.append(
            f'<div style="page-break-inside: avoid;">'
            f'<table><tr>{"".join(cells)}</tr><tr>{"".join(caps)}</tr>'
            f'<tr><td class="verdict" colspan="{len(cells)}">'
            f'<span class="{tag_cls}">Figure {n} &middot; {r["verdict_tag"]}</span> &mdash; '
            f'{r["verdict"]}</td></tr></table></div>')
    out.append("</div>")
    return "\n".join(out)


def main() -> None:
    front, sections = rc.load()
    figs = json.loads(FIGS.read_text(encoding="utf-8"))
    svg_path = print_svg()
    render_fence = fence_renderer_factory(svg_path)

    parts = []
    for s in sections:
        html = rc.restore_fences(s.html, s.blocks, render_fence)
        html = bw.align_numeric_columns(html)
        html = html.replace('<div class="tw">', "<div>")      # no overflow scrolling in print
        # a hyperlink to a local file means nothing on paper; name the file instead
        html = re.sub(r'<a href="\.\./outputs/([^"]+)">([^<]+)</a>',
                      r'<b>\2</b> <i>(outputs/\1)</i>', html)

        if s.number == 5:
            html += gallery(figs["enhancement"], 148,
                            "Enhancement, on held-out images &mdash; original, PFGT-UIE, "
                            "human reference", 2)
        if s.number == 6:
            html += gallery(figs["detection"], 200,
                            "Detection, on held-out frames &mdash; human-marked answer (left) "
                            "against what the detector found (right)", 5)

        html = substitute_glyphs(html)
        parts.append(
            f'<div class="newpage"><div class="sec">'
            f'<div class="plate">SECTION {s.number:02d}</div><h2>{s.title}</h2></div>'
            f'{html}</div>')

    toc = "".join(
        f'<tr><td class="n" width="8%">{s.number:02d}</td><td>{s.title}</td></tr>'
        for s in sections)

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><style>{CSS}</style></head><body>

<div id="footerContent">
  PFGT-UIE &mdash; Physics-Guided Frequency Transformer for Underwater Image Enhancement
  &nbsp;&middot;&nbsp; page <pdf:pagenumber> of <pdf:pagecount>
</div>

<div class="cover">
  <div class="eyebrow">UNDERWATER VISION &middot; ENGINEERING AND RESEARCH RECORD</div>
  <h1>{front['title']}</h1>
  <div class="sub">{front['subtitle']}</div>
  <div class="stand">{front['standfirst']}</div>
  <table class="spec">
    <tr>
      <td width="25%"><div class="k">HELD-OUT PSNR</div><div class="v">25.364 dB</div></td>
      <td width="25%"><div class="k">HELD-OUT SSIM</div><div class="v">0.9289</div></td>
      <td width="25%"><div class="k">DETECTION mAP@0.5</div><div class="v">0.829</div></td>
      <td width="25%"><div class="k">PARAMETERS</div><div class="v">2.31 M</div></td>
    </tr>
    <tr>
      <td colspan="4"><div class="k">ARCHITECTURAL NOVELTY</div>
        <div class="vw">None claimed &mdash; the mechanisms are published prior work.
        See Section 07.</div></td>
    </tr>
  </table>
</div>

<div class="newpage">
  <div class="plate">CONTENTS</div><h2>Contents</h2>
  <table class="toc">{toc}</table>
  <p style="margin-top:14pt; color:{INK2}; font-size:9pt;">Every figure and every number in this
  report is measured on held-out data the model was never trained on. The two interactive proof
  pages this report refers to &mdash; <i>outputs/session4_proof.html</i> and
  <i>outputs/detection_proof.html</i> &mdash; ship alongside it.</p>
</div>

{chr(10).join(parts)}
</body></html>
"""
    (BUILD / "print.html").write_text(doc, encoding="utf-8")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "wb") as fh:
        result = pisa.CreatePDF(doc, dest=fh, encoding="utf-8")

    print(f"pisa errors: {result.err}")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
