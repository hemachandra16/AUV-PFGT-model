# Is there a standard UIEB test split — and can this model be evaluated on it?

**Session 8, 2026-08-26.** Investigation only: no training, no architecture changes.

Every enhancement number this project has reported (25.364 dB and the rest) is scored on its
own seed-42 split of 89 held-out UIEB images. The report says plainly that this is not
comparable to published figures, which are quoted on what papers call "Test-U90" or "T90". This
session set out to get the comparable number.

**It could not be obtained, and the reason is worth more than the number would have been.**

---

## The short version

1. **There is no official UIEB train/test split.** The dataset's own authors publish none.
2. **There is no verifiable community-standard list either.** Across 35 repositories probed by
   840 direct file requests, exactly **one** publishes a concrete list of 90 test filenames —
   and it is a third-party benchmark aggregation, not any of the papers whose numbers we would
   be comparing against. Other repos visibly generate their *own* random splits.
3. **Even using that one list, this model cannot be scored on it.** **79 of its 90 images
   (87.8%) are in this project's training set.**
4. Scoring it anyway gives **28.19 dB** — which would look like beating U-shape Transformer by
   5.3 dB. That number is fake. It is 88% training data.
5. On the 11 images of that list this model genuinely never saw: **23.90 dB**, 95% CI
   **[21.04, 27.03]**. Too few images to claim anything, and not statistically distinguishable
   from this project's own 25.364 dB (permutation test, p = 0.349).

**The caveat in the report is not resolved. It is sharpened**: a fair comparison is still not
available, and we now know exactly what it would cost to get one.

---

## 1. What was searched

| Source | What it turned out to say |
|---|---|
| [UIEB official page](https://li-chongyi.github.io/proj_benchmark.html) (Li et al., the dataset's own authors) | 950 images, 890 with references, 60 challenging. **No train/test split is published, and no split file is offered for download.** |
| [ddz16/UIE_Benckmark](https://github.com/ddz16/UIE_Benckmark) | Ships `data/UIEB/train.txt` (800), `test.txt` (**90**), `challenging.txt` (60). A real, concrete filename list. The README does not say where the split came from or claim it is standard. |
| [LintaoPeng/U-shape-Transformer](https://github.com/LintaoPeng/U-shape-Transformer) — the 22.91 dB baseline | `utility/data.py` loads from a `list_file`, but **that list file is not in the repository** (it points at a Google Drive path). Their exact split is not published. |
| [Huang-ShiRui/Semi-UIR](https://github.com/Huang-ShiRui/Semi-UIR) | `data_split.py` **generates its own random split** — `random.seed(2022)`, 80/10/10. Not a shared list. |
| [dart-into/NMFC](https://github.com/dart-into/NMFC) | Ships `SplitTrainTest.m` and `fixedSplitTrainTest.m` — again, a paper generating its own. |
| [Li-Chongyi/Water-Net_Code](https://github.com/Li-Chongyi/Water-Net_Code) | Ships `generate_test_data.m` and a `.rar` of "a set of training and testing data" — no plain filename list. |
| 9 further repos harvested from the [community UIE list](https://github.com/lizhh268/awesome_underwater_image_enhancement-UIE-) (API sweep, `tools/_find_uieb_split.py`) | No split lists. |
| 35 repos × 12 conventional split paths × 2 branches = **840 raw-CDN probes** | **Two hits, both the same ddz16 file** under its `main` and `master` aliases. No independent second list exists to cross-check against. |

## 2. Confidence

**High confidence that no verifiable universal standard split exists.** The evidence is
consistent from three directions: the dataset authors publish nothing; the most-cited baseline
keeps its list off-repo; and two independent papers ship code that *rolls its own* split. The
"800 / 90 / 60" figures are a **size convention**, not a shared list of names. Papers reporting
"Test-U90" are, on this evidence, each reporting on their own random 90.

This matters for reading the literature: **published UIEB PSNR figures are not strictly
comparable to each other either**, since each is measured on a different 90 images. Section 4
shows that image selection alone is worth several dB.

**The one list found is internally sound but has no provenance.** `train.txt` and `test.txt`
are disjoint and their union is exactly the 890 reference pairs — it is a real, coherent split,
not a corrupted file. But one repo publishing a list is not a standard, and the brief for this
session was explicit that a fabricated consensus is worse than an honest negative. The list is
preserved at [`docs/splits/uieb_T90_ddz16.txt`](splits/uieb_T90_ddz16.txt) with its origin
recorded, and all 90 filenames were confirmed present in this project's local
`datasets/UIEB/raw-890/` and `reference-890/`.

## 3. The blocking problem: contamination

This project's seed-42 split and the ddz16 split are different random draws of the same 890
pairs, so they overlap heavily — as any two independent random splits would.

```
this project : 801 train / 89 held-out          (union 890, disjoint)
ddz16 T90    :  90 test

T90 images in this project's TRAINING set : 79 / 90   (87.8%)
T90 images in this project's held-out set : 11 / 90
T90 images in neither                     :  0 / 90
```

**Evaluating `checkpoints/best.pt` on T90 means scoring a model on its own training data for 88
of every 100 images.** That is not a weaker result; it is not a result at all.

## 4. What the numbers actually are

`tools/eval_on_list.py`, using the same model build, preprocessing and metric code as
`validate.py` — verified by reproducing this project's own figure to four decimals
(25.3644 dB).

| Group | n | PSNR | 95% CI | SSIM | UIQM | UCIQE |
|---|---|---|---|---|---|---|
| This project's own held-out split | 89 | 25.364 dB | [24.36, 26.35] | 0.9289 | 10.116 | 0.3221 |
| T90 — **all 90, 88% contaminated** | 90 | **28.186 dB** | [27.26, 29.07] | 0.9588 | 9.885 | 0.3147 |
| T90 — the 79 it was trained on | 79 | 28.783 dB | [27.93, 29.66] | 0.9648 | 9.904 | 0.3138 |
| T90 — **the 11 it genuinely never saw** | 11 | **23.901 dB** | [21.04, 27.03] | 0.9157 | 9.749 | 0.3217 |

Three things follow.

**The contaminated number is dangerously attractive.** 28.19 dB against U-shape Transformer's
published 22.91 would read as a 5.3 dB win. It is an artefact of evaluating on training data.
Anyone who downloads a split list, runs it against a model trained on a different split, and
reports the result will produce exactly this kind of number without any intent to mislead.

**The memorisation gap is +4.88 dB.** Seen 28.78 versus unseen 23.90, same model, same metric
code, same 90-image list. This is a direct measurement of how much train/test contamination
inflates UIEB PSNR, and it is larger than the entire spread between the published methods in
the report's comparison table (19.45 to 22.91 dB).

**The honest estimate is too weak to use.** 23.90 dB on 11 images has a 95% CI six dB wide, and
the gap to this project's own 25.364 dB does not survive a permutation test (p = 0.349). It
hints that the T90 images are no easier than this project's split — which is the *opposite* of
the direction the report worried about — but 11 images cannot establish that.

## 5. What a real comparison would require

Retrain the architecture from scratch on the 800-image complement of a declared test list, then
evaluate on that list. At session 3's settings (~96 epochs) and the sustained-power clamp's
~145 s/epoch, that is **roughly 4 hours** for one arm — feasible, but it is a training run, and
this session was explicitly investigation-only.

Two caveats even then. It would produce a number comparable to *one* repository's split, not to
a community standard, because section 1 establishes there is no community standard. And a
genuinely rigorous claim would need the same protocol applied to the baselines rather than
quoting their papers' self-reported figures, since those figures are each measured on their own
different 90 images.

## 6. Verdict

The report's existing caveat — *"the comparison to published methods is not valid"* — **stands,
and should be strengthened rather than removed.** The reason is now sharper than "we used a
different split": there is no standard split to have used, published UIEB numbers are not
strictly comparable to one another, and the specific comparison this project would need
requires a retrain that has not been done.

No number in this document should be quoted as this project's score on a standard benchmark.
There is no such score.
