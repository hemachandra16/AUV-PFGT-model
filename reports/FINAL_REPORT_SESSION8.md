# PFGT-UIE — Session 8: the standard split does not exist, and the number cannot be had

**2026-08-26, ~18:40 → ~19:40 local · unattended (bypass-permissions).** Investigation only:
no training, no architecture changes.

> Prior reports: [S1](FINAL_REPORT.md) · [S2](FINAL_REPORT_150EPOCH.md) ·
> [S3](FINAL_REPORT_SESSION3.md) · [S4](FINAL_REPORT_SESSION4.md) ·
> [feasibility](DATASET_EXPANSION_FEASIBILITY.md) · [S6](FINAL_REPORT_SESSION6.md) ·
> [novelty](../docs/novelty_assessment.md). Full log: [`PROGRESS.md`](PROGRESS.md).
> Full investigation: [`docs/standard_split_investigation.md`](../docs/standard_split_investigation.md).

---

## The short version

The task was to close the last open caveat: every number this project reports is scored on its
own seed-42 split of 89 UIEB images, which the report says is not comparable to published
figures quoted on "Test-U90". Find the standard split, evaluate on it, get the real number.

**There is no standard split.** The UIEB authors publish none. Across 35 repositories probed by
840 direct file requests, exactly one publishes a concrete list of 90 test filenames, and it is
a third-party benchmark aggregation with no stated provenance. The most-cited baseline in our
comparison table keeps its own list out of its repository; two other papers ship code that
*generates* their split at runtime.

**And the model cannot be scored on the one list that exists**, because **79 of its 90 images
(87.8%) are in this project's training set.**

The caveat did not close. It got stronger, and for a better reason than the one it replaces.

---

## 1. What "Test-U90" turns out to be

A size convention, not a set of images.

| Source | Finding |
|---|---|
| [UIEB official page](https://li-chongyi.github.io/proj_benchmark.html), Li et al. — the dataset's own authors | 950 images, 890 with references, 60 challenging. **No train/test split published, no split file offered.** |
| [ddz16/UIE_Benckmark](https://github.com/ddz16/UIE_Benckmark) | `train.txt` (800), `test.txt` (**90**), `challenging.txt` (60). The only concrete list found. README states no provenance and makes no standardness claim. |
| [LintaoPeng/U-shape-Transformer](https://github.com/LintaoPeng/U-shape-Transformer) — the 22.91 dB row in our table | Loads from a `list_file` that **is not in the repository** (path points at Google Drive). |
| [Huang-ShiRui/Semi-UIR](https://github.com/Huang-ShiRui/Semi-UIR) | `data_split.py` generates its own: `random.seed(2022)`, 80/10/10. |
| [dart-into/NMFC](https://github.com/dart-into/NMFC) | Ships `SplitTrainTest.m` / `fixedSplitTrainTest.m` — its own again. |
| 35 repos × 12 conventional split paths × 2 branches = **840 raw probes** | **Two hits, both the same ddz16 file** under `main` and `master`. No independent second list. |

Three independent papers each rolling their own split is positive evidence that no shared list
is in circulation — not merely a failed search.

**This has a consequence for the literature, not just for us: if every paper measures on its own
random 90 images, the published UIEB PSNR figures are not strictly comparable to each other
either.** Section 3 shows image selection alone is worth several dB.

## 2. The blocking problem

This project's seed-42 split and the ddz16 split are independent random draws of the same 890
pairs, so they overlap the way any two such draws would:

```
this project : 801 train / 89 held-out       (union 890, disjoint)
ddz16 T90    :  90 test

T90 images in this project's TRAINING set : 79 / 90   (87.8%)
T90 images in this project's held-out set : 11 / 90
T90 images in neither                     :  0 / 90
```

Evaluating `checkpoints/best.pt` on that list means scoring a model on its own training data for
88 of every 100 images. That is not a weaker result; it is not a result.

## 3. The numbers

`tools/eval_on_list.py` — same model build, preprocessing and metric code as `validate.py`,
self-validated by reproducing this project's own figure to four decimals (25.3644 dB).

| Group | n | PSNR | 95% CI | SSIM | UIQM | UCIQE |
|---|---|---|---|---|---|---|
| This project's own held-out split | 89 | 25.364 dB | [24.36, 26.35] | 0.9289 | 10.116 | 0.3221 |
| The list, **all 90 — 88% contaminated** | 90 | **28.186 dB** | [27.26, 29.07] | 0.9588 | 9.885 | 0.3147 |
| The list, the 79 it trained on | 79 | 28.783 dB | [27.93, 29.66] | 0.9648 | 9.904 | 0.3138 |
| The list, **the 11 it never saw** | 11 | **23.901 dB** | [21.04, 27.03] | 0.9157 | 9.749 | 0.3217 |

**28.19 dB is the most dangerous number this project has produced.** Against U-shape
Transformer's published 22.91 it reads as a 5.3 dB win. It is entirely an artefact of evaluating
a model on its own training data — and producing it requires no bad faith whatsoever: download a
split list, run it, report the mean. It is recorded here specifically so that nobody, including
a future session of this project, reaches for it.

**The memorisation gap is +4.88 dB** — the same model, the same metric code, the same 90-image
list, split by whether the model trained on the image. That gap is **larger than the entire
spread between every published method in the report's comparison table** (19.45 → 22.91 dB). It
is a clean incidental measurement of how much train/test contamination inflates UIEB PSNR, and
it is the one thing of value this session produced.

**The honest figure is too weak to use.** 23.901 dB on 11 images carries a 95% confidence
interval six dB wide, and its gap to this project's own 25.364 dB does not survive a two-sided
permutation test (p = 0.349). It leans towards those images being *no easier* than this
project's split — the opposite of what the report speculated might be flattering us — but eleven
images cannot establish that, and it is not claimed.

## 4. Does this change the "is it better than published work" conversation?

**It ends it, for now.** The previous answer was "we can't say, because we used a different
split." The answer now is "we can't say, because there is nothing to compare against — and
neither, strictly, can the published papers compare against each other."

The honest reading of our 25.364 dB is unchanged: **it is a number on this project's own 89
images and nothing more.** The one new piece of evidence — 23.901 dB on eleven images from
somebody else's list — is directionally *less* flattering than the headline, not more, and is
too small to matter either way. Nothing here supports a claim of beating anything.

What would a real comparison take?

1. Retrain the architecture from scratch on the 800-image complement of a declared test list,
   then evaluate on that list. ~4 h for one arm at session 3's settings under the sustained-power
   clamp. Feasible; this session was investigation-only by instruction.
2. Even then the result is comparable to *one repository's split*, not to a community standard,
   because there is no community standard.
3. A genuinely rigorous claim would additionally need the same protocol applied to the baselines
   rather than quoting their self-reported figures.

Item 3 is the expensive one and is what separates "a number" from "a comparison".

## 5. What changed in the deliverables

`docs/report_content.md` (§4, §5, §8), `docs/novelty_assessment.md` (§6) and
`PROJECT_SUMMARY.md` all previously said the comparison was blocked because this project used a
different split from the standard one. That premise was wrong. All now state the finding, and
`outputs/website.html` and `outputs/PFGT-UIE_report.pdf` were rebuilt from the corrected source
and verified by inspection — every new value present in both, every superseded phrase gone, PDF
holding at 18 pages with no orphaned pages.

**Outstanding, and outside this repository:** session 7's honest-report website was also
published as a standalone artifact at a URL held by the user. It is not produced by
`tools/build_website.py` and nothing in this session touches it. **It still carries the old,
now-incorrect claim.** The replacement text is `docs/report_content.md` §5.

## 6. Artefacts

| Path | What |
|---|---|
| [`docs/standard_split_investigation.md`](../docs/standard_split_investigation.md) | Full investigation: what was searched, what was found, confidence, and what a real comparison would require. |
| [`docs/splits/uieb_T90_ddz16.txt`](../docs/splits/uieb_T90_ddz16.txt) | The one list found, preserved with a README stating plainly that it is **not** a standard. |
| `tools/eval_on_list.py` | Scores a checkpoint on an explicit filename list, partitioned into ALL / SEEN-in-training / UNSEEN, so a contaminated number can never be reported as a test result by accident. |
| `tools/_find_uieb_split.py` | The repository sweep, kept so the negative result is reproducible. |
| `results/T90_ddz16_*.csv` | Per-image metrics for all three groups. |

**Repository state: unchanged where it matters.** `checkpoints/best.pt` is still session 3's
model at 25.364 dB, untouched. No training was run. No config was modified.
