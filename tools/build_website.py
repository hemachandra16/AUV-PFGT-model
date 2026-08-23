"""Render `docs/report_content.md` into a single self-contained page: outputs/website.html.

Design notes, since the brief asked for deliberate choices rather than defaults:

* **Palette** is taken from the two proof pages rather than invented. Amber is the colour those
  pages use for the human-marked ground truth; teal is the colour they use for the machine's
  output. Carrying that pair through the whole site means the accent colours mean something --
  amber marks human judgement and caution, teal marks machine output and confirmed results --
  instead of being decoration.
* **Type** is IBM Plex Sans Condensed for labelling (it reads as instrument and chart
  annotation, which is what the headings here are), Source Serif 4 for the body (this is a
  report, and long prose wants a text serif), IBM Plex Mono for every number. All three carry
  real fallback stacks so the page survives with no network.
* **Layout** is an asymmetric rail-plus-column, not a centred stack of cards. Square corners,
  hairline rules, no drop shadows. Section numbers hang as plate numbers.
* **Section 7 is set on a dark full-column plate** so the novelty verdict cannot be skimmed
  past, and the same verdict is repeated in the masthead spec strip above the fold.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import report_common as rc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "website.html"
FIGS = ROOT / "docs" / "_figures.json"

FONTS = ("https://fonts.googleapis.com/css2?"
         "family=IBM+Plex+Mono:wght@400;600&"
         "family=IBM+Plex+Sans+Condensed:wght@500;600;700&"
         "family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap")

# ---------------------------------------------------------------- inline SVG

PIPELINE_SVG = """
<figure class="diagram">
<svg viewBox="0 0 960 470" role="img" aria-label="PFGT-UIE enhancement pipeline: input RGB
splits into a physics prior encoder and a Haar wavelet decomposition; the LL sub-band feeds a
low-frequency transformer and the LH/HL/HH detail sub-bands feed a high-frequency transformer,
both biased by the physics priors; the two are fused, reconstructed by an inverse wavelet
transform, refined, and finally colour-corrected per image.">
  <defs>
    <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
            orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--signal)"/>
    </marker>
    <marker id="ap" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6"
            orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="var(--signal-2)"/>
    </marker>
  </defs>
  <g class="dg-flow">
    <path d="M108,180 H140"/>
    <path d="M244,180 H262 V120 H280"/>
    <path d="M244,180 H262 V240 H280"/>
    <path d="M356,120 H392"/><path d="M356,240 H392"/>
    <path d="M588,120 H614 V180 H640"/>
    <path d="M588,240 H614 V180 H640"/>
    <path d="M715,212 V356"/>
    <path d="M640,380 H612"/>
    <path d="M452,380 H424"/>
    <path d="M228,380 H200"/>
  </g>
  <g class="dg-prior">
    <path d="M62,156 V34 H120"/>
    <path d="M430,56 V88"/>
    <path d="M590,34 H636 V240 H614"/>
    <path d="M590,34 H700 V148"/>
    <path d="M590,34 H916 V444 H326 V404"/>
  </g>

  <g class="dg-box dg-prior-box">
    <rect x="120" y="12" width="470" height="44"/>
    <text class="t1" x="355" y="30">PHYSICS PRIOR ENCODER</text>
    <text class="t2" x="355" y="46">8 closed-form priors + global pool &#183; 64 ch</text>
  </g>

  <g class="dg-box">
    <rect x="16" y="156" width="92" height="48"/>
    <text class="t1" x="62" y="176">INPUT</text><text class="t2" x="62" y="192">RGB</text>

    <rect x="140" y="156" width="104" height="48"/>
    <text class="t1" x="192" y="176">HAAR DWT</text><text class="t2" x="192" y="192">1 level</text>

    <rect x="280" y="100" width="76" height="40"/>
    <text class="t1" x="318" y="118">LL</text><text class="t2" x="318" y="132">colour</text>

    <rect x="280" y="220" width="76" height="40"/>
    <text class="t1" x="318" y="238">LH HL HH</text><text class="t2" x="318" y="252">detail</text>

    <rect x="392" y="88" width="196" height="64"/>
    <text class="t1" x="490" y="112">LOW-FREQUENCY</text>
    <text class="t1" x="490" y="126">TRANSFORMER</text>
    <text class="t2" x="490" y="142">physics-guided attn &#183; d=128</text>

    <rect x="392" y="208" width="196" height="64"/>
    <text class="t1" x="490" y="232">HIGH-FREQUENCY</text>
    <text class="t1" x="490" y="246">TRANSFORMER</text>
    <text class="t2" x="490" y="262">physics-guided attn &#183; d=384</text>

    <rect x="640" y="148" width="150" height="64"/>
    <text class="t1" x="715" y="172">CROSS-FREQUENCY</text>
    <text class="t1" x="715" y="186">FUSION</text>
    <text class="t2" x="715" y="202">GroupNorm</text>

    <rect x="640" y="356" width="150" height="48"/>
    <text class="t1" x="715" y="376">INVERSE DWT</text>
    <text class="t2" x="715" y="392">full resolution</text>

    <rect x="452" y="356" width="160" height="48"/>
    <text class="t1" x="532" y="376">REFINEMENT HEAD</text>
    <text class="t2" x="532" y="392">128&#8594;64&#8594;32&#8594;3</text>

    <rect x="228" y="356" width="196" height="48"/>
    <text class="t1" x="326" y="376">GLOBAL COLOUR CORRECTION</text>
    <text class="t2" x="326" y="392">per-image affine, identity-init</text>

    <rect x="40" y="356" width="160" height="48"/>
    <text class="t1" x="120" y="376">ENHANCED</text><text class="t2" x="120" y="392">RGB</text>
  </g>

  <g class="dg-legend">
    <path class="dg-flow" d="M16,444 h26"/>
    <text class="t2" x="50" y="448" text-anchor="start">feature path</text>
    <path class="dg-prior" d="M150,444 h26"/>
    <text class="t2" x="184" y="448" text-anchor="start">physics prior</text>
  </g>
</svg>
<figcaption>The enhancement pipeline. There is deliberately no global input&#8594;output skip
connection: the network must reconstruct the image rather than learn a residual on it.</figcaption>
</figure>
"""

# Illustrative, not measured -- and labelled as such in the caption.
ATTEN_SVG = """
<figure class="atten">
<svg viewBox="0 0 300 132" role="img" aria-label="Chart of red, green and blue light intensity
falling with depth. Red is almost gone by ten metres while blue persists past thirty.">
  <g class="ax">
    <path d="M34,10 V104 H292"/>
    <text x="30" y="16" text-anchor="end">100%</text>
    <text x="30" y="107" text-anchor="end">0</text>
    <text x="34" y="120" text-anchor="start">0 m</text>
    <text x="292" y="120" text-anchor="end">30 m depth</text>
  </g>
  <path class="c-r" d="M34,10 C60,52 90,86 130,97 C180,103 240,104 292,104"/>
  <path class="c-g" d="M34,10 C80,26 150,54 210,72 C240,81 270,87 292,91"/>
  <path class="c-b" d="M34,10 C90,20 160,38 220,52 C250,59 275,64 292,68"/>
  <text class="lb c-rt" x="136" y="92">R</text>
  <text class="lb c-gt" x="216" y="68">G</text>
  <text class="lb c-bt" x="222" y="44">B</text>
</svg>
<figcaption>Why underwater photographs go blue. Illustrative curves using typical coastal-water
attenuation coefficients &#8212; red is largely gone within a few metres, blue persists. The
network's R/G and R/B priors read this as an effective depth cue.</figcaption>
</figure>
"""

# ---------------------------------------------------------------- css

CSS = """
:root{
  color-scheme:light dark;
  --paper:#e9eef0; --surface:#f5f8f9; --ink:#0d1a1d; --ink-2:#4c6065;
  --rule:#bfcdd0; --rule-soft:#d6e0e2;
  --signal:#0b6d5c; --signal-2:#8a6108;
  --abyss:#05100f; --abyss-ink:#e8f2f0; --abyss-dim:#7c9899; --abyss-rule:#1c3230;
  --font-display:"IBM Plex Sans Condensed","Roboto Condensed","Arial Narrow",system-ui,sans-serif;
  --font-body:"Source Serif 4","Source Serif Pro",Georgia,"Times New Roman",serif;
  --font-mono:"IBM Plex Mono",ui-monospace,"Cascadia Mono",Consolas,monospace;
}
@media (prefers-color-scheme:dark){
  :root{
    --paper:#070f10; --surface:#0d1a1c; --ink:#e2ecec; --ink-2:#8aa1a4;
    --rule:#1e3134; --rule-soft:#16272a;
    --signal:#3ad7b3; --signal-2:#ffd044;
    --abyss:#030b0c; --abyss-ink:#e8f2f0; --abyss-dim:#7c9899; --abyss-rule:#193230;
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);
     font-family:var(--font-body);font-size:17px;line-height:1.62;
     font-optical-sizing:auto;text-rendering:optimizeLegibility}
a{color:inherit;text-decoration:none;border-bottom:1px solid var(--signal);
  padding-bottom:1px;transition:background .12s}
a:hover{background:color-mix(in srgb,var(--signal) 16%,transparent)}
h2,h3,h4{font-family:var(--font-display);font-weight:600;line-height:1.18;letter-spacing:-.005em}
strong{font-weight:600}
code{font-family:var(--font-mono);font-size:.86em;background:var(--surface);
     border:1px solid var(--rule-soft);padding:.05em .3em}

/* ---------- masthead ---------- */
/* The masthead and the section-7 plate are dark in BOTH themes, so they must not inherit the
   darkened light-mode accents -- on a near-black ground those drop to ~3.5:1. Re-declare the
   accent tokens locally; everything inside then resolves against the dark ground it sits on. */
.mast,#s7{--signal:#3ad7b3;--signal-2:#ffd044;--rule:var(--abyss-rule)}
.mast{background:var(--abyss);color:var(--abyss-ink);
      border-bottom:1px solid var(--abyss-rule)}
.mast-in{max-width:1180px;margin:0 auto;padding:44px 32px 0}
.eyebrow{font-family:var(--font-mono);font-size:.66rem;letter-spacing:.24em;
         text-transform:uppercase;color:var(--signal)}
.mast h1{font-family:var(--font-display);font-weight:700;letter-spacing:-.02em;
         font-size:clamp(3rem,11vw,7rem);line-height:.9;margin:.18em 0 0}
.mast .sub{font-size:clamp(1rem,2.2vw,1.32rem);color:var(--abyss-ink);
           max-width:34ch;margin:.7em 0 0;line-height:1.32}
.mast .stand{color:var(--abyss-dim);max-width:60ch;margin:1.1em 0 0;font-size:.96rem}
.mast .stand strong{color:var(--abyss-ink);font-weight:400}
.mast-grid{display:grid;grid-template-columns:minmax(0,1fr) 300px;gap:40px;align-items:end}
@media(max-width:900px){.mast-grid{grid-template-columns:1fr}}

.spec{display:flex;flex-wrap:wrap;gap:0;margin:40px -32px 0;
      border-top:1px solid var(--abyss-rule)}
.spec div{flex:1 1 150px;padding:16px 32px;border-right:1px solid var(--abyss-rule)}
.spec div:last-child{border-right:0;flex:1 1 250px}
.spec dt{font-family:var(--font-display);font-size:.64rem;letter-spacing:.16em;
         text-transform:uppercase;color:var(--abyss-dim)}
.spec dd{margin:.3em 0 0;font-family:var(--font-mono);font-size:1.28rem;color:var(--signal)}
.spec dd small{display:block;font-family:var(--font-body);font-size:.72rem;
               color:var(--abyss-dim);letter-spacing:0;margin-top:.2em}
.spec .warn dd{color:var(--signal-2);font-size:1.02rem;line-height:1.2}

.atten svg{width:100%;height:auto;display:block}
.atten .ax path{fill:none;stroke:var(--abyss-rule);stroke-width:1}
.atten .ax text{font-family:var(--font-mono);font-size:7px;fill:var(--abyss-dim)}
.atten path[class^=c-]{fill:none;stroke-width:2}
.atten .c-r{stroke:#e0603f}.atten .c-g{stroke:#4fb477}.atten .c-b{stroke:#4b8fe0}
.atten .lb{font-family:var(--font-display);font-weight:700;font-size:11px}
.atten .c-rt{fill:#e0603f}.atten .c-gt{fill:#4fb477}.atten .c-bt{fill:#4b8fe0}
.atten figcaption{color:var(--abyss-dim);font-size:.72rem;line-height:1.45;
                  margin-top:.5em;font-family:var(--font-body)}
.mast figure{margin:0 0 8px}

/* ---------- rail + column ---------- */
.wrap{max-width:1180px;margin:0 auto;padding:0 32px;
      display:grid;grid-template-columns:206px minmax(0,1fr);gap:0 56px}
.rail{position:sticky;top:0;align-self:start;padding:44px 0;
      max-height:100vh;overflow-y:auto}
.rail h6{font-family:var(--font-display);font-size:.62rem;letter-spacing:.18em;
         text-transform:uppercase;color:var(--ink-2);margin:0 0 .9em;font-weight:600}
.rail ol{list-style:none;margin:0;padding:0;counter-reset:r}
.rail li{counter-increment:r;margin:0 0 2px}
.rail a{display:block;border:0;padding:3px 0 3px 26px;position:relative;
        font-family:var(--font-display);font-size:.86rem;line-height:1.28;color:var(--ink-2)}
.rail a::before{content:counter(r,decimal-leading-zero);position:absolute;left:0;top:4px;
        font-family:var(--font-mono);font-size:.66rem;color:var(--rule)}
.rail a:hover,.rail a.on{color:var(--ink);background:transparent}
.rail a.on::before{color:var(--signal)}
.rail a.flag{color:var(--signal-2)}
.rail .out{margin-top:1.6em;padding-top:1.1em;border-top:1px solid var(--rule)}
.rail .out a{padding-left:0;font-size:.8rem;color:var(--signal)}
@media(max-width:860px){
  .wrap{grid-template-columns:1fr;gap:0}
  .rail{position:static;max-height:none;padding:24px 0 0;
        border-bottom:1px solid var(--rule);margin-bottom:20px}
  .rail ol{display:flex;flex-wrap:wrap;gap:0 20px}
}

main{padding:44px 0 0;min-width:0}
section{padding:0 0 62px;border-bottom:1px solid var(--rule);margin-bottom:52px}
section:last-of-type{border-bottom:0}
section>h2{font-size:clamp(1.7rem,3.6vw,2.5rem);margin:0 0 1rem;max-width:22ch}
.plate{font-family:var(--font-mono);font-size:.7rem;letter-spacing:.2em;color:var(--signal);
       display:block;margin-bottom:.7em}
main p{max-width:66ch;margin:0 0 1.05em}
main h3{font-size:1.16rem;margin:2.4em 0 .6em;max-width:44ch}
main h4{font-size:.96rem;margin:1.8em 0 .5em;color:var(--ink-2);
        text-transform:uppercase;letter-spacing:.1em;font-size:.72rem}
main ul,main ol{max-width:64ch;padding-left:1.2em}
main li{margin:0 0 .5em}
blockquote{margin:1.3em 0;padding:.85rem 0 .85rem 1.15rem;
           border-left:3px solid var(--signal-2);background:var(--surface)}
blockquote p{margin:0;max-width:62ch;padding-right:1rem}

/* ---------- tables ---------- */
.tw{overflow-x:auto;margin:1.6em 0;border-top:1px solid var(--ink)}
table{width:100%;border-collapse:collapse;font-family:var(--font-mono);
      font-size:.78rem;font-variant-numeric:tabular-nums}
th{font-family:var(--font-display);font-weight:600;font-size:.68rem;letter-spacing:.09em;
   text-transform:uppercase;text-align:left;color:var(--ink-2);
   padding:.6rem .8rem;border-bottom:1px solid var(--rule);white-space:nowrap}
td{padding:.52rem .8rem;border-bottom:1px solid var(--rule-soft);vertical-align:top}
tr:last-child td{border-bottom:1px solid var(--ink)}
.ar{text-align:right}
td strong,th strong{color:var(--signal)}
tbody tr:hover td{background:var(--surface)}

/* ---------- fenced blocks ---------- */
pre{font-family:var(--font-mono);font-size:.73rem;line-height:1.55;overflow-x:auto;
    background:var(--surface);border:1px solid var(--rule-soft);border-left:3px solid var(--rule);
    padding:.9rem 1rem;margin:1.5em 0}
.formula{font-family:var(--font-mono);font-size:1.06rem;text-align:center;
         padding:1.15rem;margin:1.5em 0;color:var(--signal);
         border-top:1px solid var(--rule);border-bottom:1px solid var(--rule)}

/* ---------- diagram ---------- */
figure{margin:2em 0}
.diagram svg{width:100%;height:auto;display:block;background:var(--surface);
             border:1px solid var(--rule-soft);padding:8px 0}
.dg-flow path{fill:none;stroke:var(--signal);stroke-width:1.4;marker-end:url(#ah)}
.dg-prior path{fill:none;stroke:var(--signal-2);stroke-width:1.2;stroke-dasharray:4 3;
               marker-end:url(#ap)}
.dg-legend path{marker-end:none}
.dg-box rect{fill:var(--paper);stroke:var(--ink);stroke-width:1.2}
.dg-prior-box rect{stroke:var(--signal-2);stroke-dasharray:5 3;fill:none}
.dg-box text,.dg-legend text{text-anchor:middle;fill:var(--ink)}
.dg-box .t1{font-family:var(--font-display);font-weight:600;font-size:10.5px;
            letter-spacing:.06em}
.dg-box .t2,.dg-legend .t2{font-family:var(--font-mono);font-size:8px;fill:var(--ink-2)}
figcaption{font-size:.82rem;color:var(--ink-2);line-height:1.5;margin-top:.7em;max-width:62ch}

/* ---------- galleries ---------- */
.gal{margin:2em 0;border-top:1px solid var(--ink)}
.gal .row{padding:1.2rem 0;border-bottom:1px solid var(--rule-soft)}
.gal .strip{display:grid;gap:2px;grid-template-columns:repeat(3,minmax(0,1fr))}
.gal .strip.two{grid-template-columns:repeat(2,minmax(0,1fr))}
.gal figure{margin:0}
.gal img{width:100%;height:auto;display:block;background:var(--abyss)}
.gal figcaption{font-family:var(--font-display);font-size:.64rem;letter-spacing:.11em;
                text-transform:uppercase;color:var(--ink-2);margin:.45em 0 0}
.gal .verdict{margin:.8rem 0 0;font-size:.9rem;max-width:74ch;color:var(--ink-2)}
.gal .verdict b{color:var(--ink);font-weight:400}
.tag{display:inline-block;font-family:var(--font-mono);font-size:.62rem;letter-spacing:.12em;
     border:1px solid var(--signal);color:var(--signal);padding:.14em .5em;margin-right:.7em;
     vertical-align:2px}
.tag.w{border-color:var(--signal-2);color:var(--signal-2)}
.legend{font-family:var(--font-display);font-size:.68rem;letter-spacing:.1em;
        text-transform:uppercase;color:var(--ink-2);display:flex;gap:1.5rem;flex-wrap:wrap;
        padding:.6rem 0}
.legend i{display:inline-block;width:10px;height:10px;margin-right:.4em;vertical-align:-1px}
@media(max-width:620px){.gal .strip{grid-template-columns:1fr}}

/* ---------- journey ---------- */
#s4 .tl{border-left:1px solid var(--rule);padding-left:30px;margin-left:5px}
#s4 .tl h3{position:relative;margin-top:2.2em}
#s4 .tl h3:first-child{margin-top:.4em}
#s4 .tl h3::before{content:"";position:absolute;left:-35px;top:.52em;
                   width:9px;height:9px;background:var(--paper);
                   border:2px solid var(--signal)}
#s4 .tl blockquote{border-left-color:var(--signal-2)}
#s4 .tl p{max-width:62ch}

/* ---------- novelty plate ---------- */
#s7{background:var(--abyss);color:var(--abyss-ink);padding:44px 36px 40px;
    border:1px solid var(--abyss-rule);border-bottom:1px solid var(--abyss-rule)}
#s7 .plate{color:var(--signal-2)}
#s7 h2,#s7 h3{color:var(--abyss-ink)}
#s7 p{color:var(--abyss-dim)}
#s7 strong{color:var(--abyss-ink);font-weight:600}
#s7 em{color:var(--abyss-ink)}
#s7 code{background:transparent;border-color:var(--abyss-rule);color:var(--signal-2)}
#s7 a{border-bottom-color:var(--signal-2)}
#s7 blockquote{background:transparent;border-left:0;border-top:2px solid var(--signal-2);
               border-bottom:2px solid var(--signal-2);padding:1.1rem 0;margin:0 0 1.6em}
#s7 blockquote p{font-family:var(--font-display);font-weight:700;color:var(--signal-2);
                 font-size:clamp(1.5rem,4.4vw,2.5rem);line-height:1.08;letter-spacing:-.015em;
                 max-width:20ch}
#s7 blockquote strong{color:var(--signal-2);font-weight:700}

/* ---------- references ---------- */
#s9 ol{max-width:78ch;padding-left:2.2em;font-size:.92rem}
#s9 li{margin-bottom:.85em}
#s9 a{color:var(--signal);border-bottom-color:var(--rule)}

footer{border-top:1px solid var(--rule);margin-top:20px}
.foot-in{max-width:1180px;margin:0 auto;padding:26px 32px 60px;
         font-family:var(--font-mono);font-size:.7rem;color:var(--ink-2);
         display:flex;justify-content:space-between;gap:1.5rem;flex-wrap:wrap}
"""

JS = """
(function(){
  var links=[].slice.call(document.querySelectorAll('.rail a[href^="#"]'));
  var secs=links.map(function(a){return document.getElementById(a.getAttribute('href').slice(1))})
                .filter(Boolean);
  if(!('IntersectionObserver' in window)||!secs.length)return;
  function mark(){
    var best=0,bd=1e9;
    secs.forEach(function(s,i){var d=Math.abs(s.getBoundingClientRect().top-90);
                               if(d<bd){bd=d;best=i}});
    links.forEach(function(a,i){a.classList.toggle('on',i===best)});
  }
  addEventListener('scroll',mark,{passive:true});addEventListener('resize',mark);mark();
})();
"""

# ---------------------------------------------------------------- helpers

NUMERIC = re.compile(r"^[\s—−+-]*[\d.,]+\s*(dB|M|%|ch|m|px)?\s*$|^\s*[—-]\s*$")


def align_numeric_columns(html: str) -> str:
    """Right-align table columns whose data cells are predominantly numeric.

    Underwater results tables mix a label column with several measurement columns; left-aligned
    decimals are hard to compare down a column, and blanket right-alignment mangles the prose
    tables in section 2. So decide per column, from the data.
    """
    def fix_table(m):
        tbl = m.group(0)
        rows = re.findall(r"<tr>(.*?)</tr>", tbl, re.S)
        if not rows:
            return tbl
        body = [re.findall(r"<td[^>]*>(.*?)</td>", r, re.S) for r in rows]
        body = [b for b in body if b]
        if not body:
            return tbl
        ncol = max(len(b) for b in body)
        numeric = []
        for c in range(ncol):
            cells = [re.sub(r"<[^>]+>", "", b[c]).strip() for b in body if len(b) > c]
            cells = [x for x in cells if x]
            numeric.append(bool(cells) and sum(bool(NUMERIC.match(x)) for x in cells) >= 0.7 * len(cells))

        def redo(rm):
            row, idx = rm.group(1), [0]
            def cell(cm):
                tag, attrs, inner = cm.group(1), cm.group(2), cm.group(3)
                i = idx[0]; idx[0] += 1
                if i < ncol and numeric[i]:
                    attrs = ' class="ar"' + attrs
                return f"<{tag}{attrs}>{inner}</{tag}>"
            return "<tr>" + re.sub(r"<(td|th)([^>]*)>(.*?)</\1>", cell, row, flags=re.S) + "</tr>"

        return re.sub(r"<tr>(.*?)</tr>", redo, tbl, flags=re.S)

    html = re.sub(r"<table>.*?</table>", fix_table, html, flags=re.S)
    return re.sub(r"(<table>.*?</table>)", r'<div class="tw">\1</div>', html, flags=re.S)


def fence_renderer(lang: str, body: str) -> str:
    if lang == "math":
        return f'<div class="formula">{body.strip()}</div>'
    if "Haar DWT" in body and "─" in body:          # the ASCII pipeline -> real diagram
        return PIPELINE_SVG
    esc = body.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"<pre>{esc.rstrip()}</pre>"


def gallery(rows, legend_html="") -> str:
    out = ['<div class="gal">']
    if legend_html:
        out.append(f'<div class="legend">{legend_html}</div>')
    for r in rows:
        two = " two" if len(r["panels"]) == 2 else ""
        out.append(f'<div class="row"><div class="strip{two}">')
        for p in r["panels"]:
            out.append(f'<figure><img src="{p["src"]}" alt="{p["label"]}" loading="lazy">'
                       f'<figcaption>{p["label"]}</figcaption></figure>')
        out.append("</div>")
        tag_cls = "tag w" if r["verdict_tag"].upper() in ("WORSE", "TURTLE", "CORALS",
                                                          "SCALLOP", "HOLOTHURIAN") else "tag"
        out.append(f'<p class="verdict"><span class="{tag_cls}">{r["verdict_tag"]}</span>'
                   f'<b>{r["verdict"]}</b></p></div>')
    out.append("</div>")
    return "\n".join(out)


def main() -> None:
    front, sections = rc.load()
    figs = json.loads(FIGS.read_text(encoding="utf-8"))

    body_parts = []
    for s in sections:
        html = rc.restore_fences(s.html, s.blocks, fence_renderer)
        html = align_numeric_columns(html)
        # proof pages ship as siblings of website.html, so drop the docs/-relative prefix
        html = html.replace('href="../outputs/', 'href="')

        if s.number == 4:                       # research journey -> timeline rail
            html = f'<div class="tl">{html}</div>'

        if s.number == 5:                        # enhancement gallery after the results tables
            html += gallery(figs["enhancement"])

        if s.number == 6:                        # detection gallery, with its box-colour key
            key = ('<span><i style="background:var(--signal-2)"></i>human-marked answer</span>'
                   '<span><i style="background:var(--signal)"></i>what the detector found</span>')
            html += gallery(figs["detection"], key)

        title_html = s.title.replace("--", "&#8212;")
        body_parts.append(
            f'<section id="s{s.number}">'
            f'<h2><span class="plate">Section {s.number:02d}</span>{title_html}</h2>'
            f'{html}</section>')

    nav = "\n".join(
        f'<li><a href="#s{s.number}"{" class=flag" if s.number == 7 else ""}>{s.title}</a></li>'
        for s in sections)

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PFGT-UIE &#8212; Physics-Guided Frequency Transformer for Underwater Image Enhancement</title>
<meta name="description" content="A seven-session engineering and research record: 25.364 dB
held-out enhancement, mAP@0.5 0.829 detection, and an honest account of what is and is not novel.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{CSS}</style>
</head>
<body>

<header class="mast">
  <div class="mast-in">
    <div class="mast-grid">
      <div>
        <p class="eyebrow">Underwater vision &#183; engineering and research record</p>
        <h1>{front['title']}</h1>
        <p class="sub">{front['subtitle']}</p>
        <div class="stand">{front['standfirst']}</div>
      </div>
      {ATTEN_SVG}
    </div>
    <dl class="spec">
      <div><dt>Held-out PSNR</dt><dd>25.364 dB<small>89 UIEB images, never trained on</small></dd></div>
      <div><dt>Held-out SSIM</dt><dd>0.9289<small>same split</small></dd></div>
      <div><dt>Detection mAP@0.5</dt><dd>0.829<small>4,200 RUOD images, 10 classes</small></dd></div>
      <div><dt>Parameters</dt><dd>2.31 M<small>enhancer; detector 2.6 M</small></dd></div>
      <div class="warn"><dt>Architectural novelty</dt><dd>None claimed<small>the mechanisms are
        published prior work &#8212; see Section 07</small></dd></div>
    </dl>
  </div>
</header>

<div class="wrap">
  <nav class="rail">
    <h6>Contents</h6>
    <ol>{nav}</ol>
    <div class="out">
      <h6>See it yourself</h6>
      <p><a href="session4_proof.html">Enhancement proof</a></p>
      <p><a href="detection_proof.html">Detection proof</a></p>
    </div>
  </nav>
  <main>
{chr(10).join(body_parts)}
  </main>
</div>

<footer><div class="foot-in">
  <span>PFGT-UIE &#183; every figure on this page is from held-out data</span>
  <span>Rendered from docs/report_content.md</span>
</div></footer>

<script>{JS}</script>
</body>
</html>
"""
    OUT.write_text(page, encoding="utf-8")
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KB)")
    print(f"  sections   : {len(sections)}")
    print(f"  tables     : {page.count('<table>')}")
    print(f"  images     : {page.count('<img ')}")
    print(f"  inline svg : {page.count('<svg ')}")


if __name__ == "__main__":
    main()
