# Dataset expansion feasibility — LSUI and EUVP

**2026-08-22 · verification only. No training was run, no model/config/training code changed.**

## Verdict

| Dataset | Verdict | New pairs | Leak risk to the held-out 89 |
|---|---|---|---|
| **LSUI** | **PROCEED** | ~4,277 | **None found** — 0 duplicates, closest match 8 bits away |
| **EUVP** (`underwater_dark`) | **PROCEED WITH CAVEATS** | ~5,548 | **None found** — 0 duplicates, closest match 8 bits away |
| EUVP (`underwater_imagenet`) | **DON'T** | 3,700 | not checked — synthetic, see §7 |

Both are fetchable with **zero authentication**, both were **actually downloaded and verified**
(not just link-checked), and neither contains any of UIEB's held-out test images. Combined,
they would take the training pool from **801 → ~10,600 pairs, a 13× increase**, for under
1.5 GB of disk and about a minute of download.

**The one thing that would make me hesitate is not access or leakage — it is §6.** The three
datasets disagree measurably about what "enhanced" means, and EUVP disagrees most. Read §6
before committing to a run, and use the protocol in §8.

---

## 1. Can they be fetched without a login or a form?

**LSUI — yes.** The official GitHub offers BaiduYun (needs a Chinese account — a hard blocker
for unattended use) **and Google Drive, which is public and needs nothing**:

```
https://drive.google.com/file/d/10gD4s12uJxCHcuFdX9Khkv37zzBwNFbL   -> LSUI.zip
```
Probed the Drive endpoint directly, followed the virus-scan interstitial, and read the real
response headers:
```
interstitial: filename=LSUI.zip  size=470M   confirm token present: True
AFTER CONFIRM: content-type=application/octet-stream  content-length=492658225  (0.49 GB)
first 4 bytes: b'PK\x03\x04'          <- a genuine zip, not an HTML error page
```

**EUVP — yes, but not from the official source.** `irvlab.cs.umn.edu/resources/euvp-dataset`
returns **HTTP 403** to automated fetching. However a public Hugging Face mirror works with no
auth, exactly like the RUOD precedent from session 1:

```
Ken1053/EUVP  -> data/train-00000-of-00001.parquet   189.3 MB, 5,550 rows
```
5,550 is exactly the size of EUVP's `underwater_dark` paired subset, so this is that subset.

**Blockers found and NOT worked around** (flagged rather than circumvented, as instructed):
* BaiduYun — needs an account; not attempted.
* Kaggle (`noureldin199/lsui-...`) — needs API credentials; not attempted. Not needed, since
  the Drive link works.
* OpenDataLab — needs login; not attempted.
* `Gusanagy/LSUI-TURBID-UIEB` on HF — **empty**, contains only `.gitattributes`. Dead end
  despite appearing in search results.

## 2. Access proven, not assumed

Both were downloaded in full and opened:

```
_dsprobe/LSUI.zip          492.7 MB in 23 s   (~21 MB/s)
_dsprobe/euvp_dark.parquet 189.3 MB in 21 s
```
LSUI.zip opens as a valid archive with 8,558 image entries; the parquet decodes to real JPEGs.

## 3. Format and resolution compatibility

`data/dataset.py` pairs images by **identical filename in the raw and reference directories**
(`raw_names & reference_names`). Nothing else matters — extension, resolution and format are
all handled downstream, since `_load_image` resizes everything to 256×256.

**LSUI needs essentially no conversion:**
```
LSUI/input/N.jpg   4,279 files
LSUI/GT/N.jpg      4,279 files
basename intersection: 4279 / 4279 / 4279     <- a perfect 1:1 pairing already
```
Unzip, point `raw_dir` at `LSUI/input` and `reference_dir` at `LSUI/GT`. That is the whole job.

**EUVP needs a small extraction step.** The parquet holds embedded images
(`input_image`, `edited_image`, both 256×256 JPEG). Writing them out as
`euvp/input/N.jpg` + `euvp/GT/N.jpg` is roughly 15 lines. Verified the pairs are correctly
oriented — input is cast, edited is corrected — both by eye and numerically:
```
pair1: cast 0.842 -> 0.134    pair2: cast 1.195 -> 0.179
```

**Resolutions** (300 LSUI samples): 320×240 (39%), 640×360 (22%), 400×300 (15%), 256×256 (14%),
up to 1280×1024. All RGB. EUVP is uniformly 256×256. Both are **lower resolution than UIEB**
(which reaches 1800×1295) — irrelevant at the current 256×256 training size, but it means this
extra data would give no benefit if training resolution is ever raised.

**Filename collisions:** LSUI uses bare integers (`0.jpg`), UIEB uses `NNN_img_.png` and some
bare integers (`5554.png`). Extensions differ so a merge would not actually collide, but a
per-source prefix (`lsui_0.jpg`) is cheap insurance and recommended.

## 4. Leakage against the held-out 89 — the critical check

**Method.** Checksums are useless here: any shared image would have been resized and
re-encoded. Used 64-bit **dHash** + **aHash**, threshold ≤6 bits for a confident duplicate,
re-checked with normalised 32×32 grayscale correlation, and additionally hashed a
**mirrored** copy of every held-out image so a horizontally flipped duplicate could not slip
through. All 4,279 LSUI and all 5,550 EUVP images were compared against all 890 UIEB images —
not a sample.

**The method was validated with a positive control before its negative result was trusted.**
It found genuine duplicates against UIEB's *training* split:
```
LSUI/input/2061.jpg  <-> 917_img_.png   1 bit,  corr 0.995
euvp_dark/151        <-> 922_img_.png   3 bits, corr 0.998
```
Rendered side by side, `LSUI/input/2061` vs `917_img_` is unmistakably the same photograph —
same flounder, same coral, same fish in the top-left corner. So the detector demonstrably
catches real duplicates.

**Result against the held-out 89: clean, for both.**

```
LSUI   scanned 4279   closest held-out match: 8 bits   confident duplicates: 0
EUVP   scanned 5550   closest held-out match: 8 bits   confident duplicates: 0
```

For contrast, confirmed duplicates sit at 1–3 bits. The closest held-out candidate
(`LSUI/input/1145` vs `711_img_`, 9 bits, corr 0.84) was inspected visually and is **a clearly
different scene** — both are blue-green reef shots, which is all the correlation reflects.

**Total overlap with UIEB is tiny:** only **2 of 890** UIEB images have a confident duplicate in
LSUI, and **2 of 890** in EUVP — all four in the training split. So ~4,277 and ~5,548 images
respectively are genuinely new.

> **Mandatory for the future session:** these datasets *do* demonstrably share source material
> with UIEB, so a clean held-out result is partly luck. Re-run
> `python tools/check_dataset_leakage.py` after any change to the split, the datasets, or the
> seed. The tool is committed for exactly this purpose. **Do not treat this report as a
> standing guarantee.**

## 5. Size and time budget

Free space before anything: **118 GB**. This is a rounding error against that.

| Item | Download | On disk after extraction |
|---|---|---|
| LSUI.zip | 0.49 GB | 0.50 GB |
| EUVP `underwater_dark` parquet | 0.19 GB | ~0.35 GB as loose JPEGs |
| **Total** | **0.68 GB** | **~0.85 GB** |

Measured throughput was ~21 MB/s, so **fetching both takes about 45 seconds**. Unzipping LSUI
and extracting the parquet is a few minutes of CPU. Preprocessing beyond that is zero for LSUI
and near-zero for EUVP.

Training cost is the real expense, not the data: the pool grows 801 → ~10,600, so an epoch
becomes ~13× longer. At the ~50 s/epoch seen when the GPU is unthrottled that is ~11 min/epoch;
at the ~146 s/epoch seen when it is throttled it is ~32 min/epoch. **Sort the GPU throttling
out before attempting this** — see prior sessions' reports.

## 6. The real risk: the datasets disagree about what "enhanced" means

Session 4 established that a large part of UIEB's reference colour is the human retoucher's
choice rather than a recoverable property of the input (held-out R² of 0.015/0.104/0.346).
That makes reference *style* a first-class concern when mixing datasets. Measured over 400
reference images from each:

| target set | R | G | B | R/G | R/B | cast | saturation |
|---|---|---|---|---|---|---|---|
| UIEB reference | 0.368 | 0.459 | 0.478 | 0.806 | **0.807** | 0.345 | 0.226 |
| LSUI GT | 0.389 | 0.435 | 0.425 | 0.906 | **0.967** | 0.269 | 0.175 |
| EUVP edited | 0.407 | 0.421 | 0.389 | 0.981 | **1.154** | 0.374 | 0.186 |

UIEB's idea of "corrected" is still slightly blue-leaning (R/B 0.81). LSUI's is warmer (0.97).
**EUVP's is warmer still — red actually exceeds blue (1.15), a R/B divergence of +0.35.**

Training on a naive union teaches the model to aim at a warmer target than the held-out UIEB
references reward, and the evaluation is entirely against UIEB. That is a plausible mechanism
for *more data making the headline number worse* — which is exactly the sort of surprise this
verification pass exists to prevent. LSUI is the closer of the two; EUVP is the riskier.

This is a hypothesis from measured statistics, not a demonstrated effect — but it is cheap to
neutralise, and §8 does.

## 7. Licensing

**LSUI — fine for academic research.** The project repository is MIT-licensed; the dataset
page states the images "have been licensed and used only for academic purposes" and asks that
the paper be cited. Cite Peng, Zhu & Bian, *U-shape Transformer for Underwater Image
Enhancement*, IEEE TIP 2023 (arXiv:2111.11843).

**EUVP — almost certainly fine, but not confirmed from the primary source.** The official
IRVLab page returned 403, so its terms could not be read directly. The dataset is a standard
academic release accompanying Islam, Xia & Sattar, *Fast Underwater Image Enhancement for
Improved Visual Perception*, IEEE RA-L 2020 (arXiv:1903.09766), and one HF mirror of a sibling
subset is tagged Apache-2.0. **Recommend a human confirms the IRVLab terms in a browser before
publication**, and cite the paper regardless. This is the one genuinely unresolved item in the
report.

**`underwater_imagenet` — recommend skipping** on top of the licensing point: that subset is
*synthetically* degraded ImageNet-derived imagery, not real underwater photographs. It is a
domain mismatch for a physics-guided model whose whole premise is real attenuation and
scattering.

## 8. If a future session proceeds, do exactly this

1. **Fetch** (~45 s):
   LSUI via the Drive id `10gD4s12uJxCHcuFdX9Khkv37zzBwNFbL` (follow the confirm-token flow —
   a plain GET returns the HTML interstitial, not the zip);
   EUVP via `Ken1053/EUVP`'s parquet.
2. **Lay out** as `datasets/LSUI/{input,GT}` and `datasets/EUVP/{input,GT}`, prefixing
   filenames per source (`lsui_`, `euvp_`) to remove any chance of collision.
3. **Re-run `python tools/check_dataset_leakage.py` and require 0 confident held-out hits.**
   Treat a nonzero result as a stop, not a warning.
4. **Keep the evaluation untouched.** `data/dataset.py::get_splits()` must continue to produce
   the identical 89 UIEB held-out images with seed 42, so every number stays comparable to
   sessions 1–4. Extra data goes into the *training* pool only. Verify by asserting the
   held-out filename list is unchanged before training starts.
5. **Pretrain on the union, then fine-tune on UIEB-train alone.** This is the fix for §6: the
   model gets the 13× data benefit for learning underwater structure, but *finishes* on UIEB's
   colour convention, which is what the held-out set scores. A naive single-stage union run is
   the riskier design.
6. **Budget the GPU first.** 13× the data per epoch. Fix the power throttling before starting,
   or reduce epochs accordingly.

**Recommended scope for a first attempt:** LSUI only. It is leak-free, needs no conversion, is
stylistically closer to UIEB, and still gives a 6.3× larger training pool. Add EUVP as a second
experiment once LSUI's effect is known — otherwise two variables change at once, which is the
attribution problem sessions 3 and 4 already ran into.

---

## Evidence and artefacts

| What | Where |
|---|---|
| Leak-check tool (committed, re-runnable) | `tools/check_dataset_leakage.py` |
| Downloaded candidates + rendered comparisons | `_dsprobe/` (gitignored, ~680 MB) |
| Positive controls proving the detector works | `_dsprobe/heldout_hits/POSITIVE_CONTROL_*.png` |
| Closest held-out candidates (all benign) | `_dsprobe/heldout_hits/` |

Time spent: about one hour, most of it on the leakage check. Nothing about these two datasets
turned out to be harder to work with than expected — the only surprise was the reference-style
divergence in §6, which is a modelling consideration rather than an access problem.
