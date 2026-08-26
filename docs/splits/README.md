# Split lists

`uieb_T90_ddz16.txt` — 90 UIEB filenames, fetched 2026-08-26 from
`https://raw.githubusercontent.com/ddz16/UIE_Benckmark/main/data/UIEB/test.txt`.

**This is not an official or verified-standard split.** It is the only concrete UIEB test-set
filename list found in a public repository, and it carries no stated provenance. The UIEB
authors publish no train/test split at all, and other papers visibly generate their own. See
[`../standard_split_investigation.md`](../standard_split_investigation.md).

It is kept here so the numbers in that investigation are reproducible, not because it should be
treated as a benchmark. 79 of its 90 images are in this project's training set, so
`checkpoints/best.pt` cannot be honestly scored on it.
