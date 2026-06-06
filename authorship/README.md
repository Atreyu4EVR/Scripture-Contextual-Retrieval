# Book of Mormon Authorship / Production Study — Phase One

This workstream applies computational linguistics and the scientific method to the
question: *Which natural-language production model best explains the observable textual
features of the Book of Mormon?* It tests four hypotheses — H1 single-author 19th-century
composition, H2 19th-century source/collaboration, H3 internal multi-voice, H4
translation-mediated text.

**Phase one (this directory)** addresses the **internal** hypotheses **H3 and H4** using
only texts already in the repository (the Book of Mormon plus the KJV Old/New Testament as
a biblical-quotation control and as a known-authorship calibration corpus). It deliberately
does **not** address H1/H2 — comparison against Joseph Smith, Rigdon, Spalding, Ethan Smith,
and distractor authors requires an external corpus and is reserved for phase two.

## What it does

1. **Rule-based annotation** of every chapter by claimed narrator, editorial layer
   (small-plates self vs Mormon/Moroni abridgement), embedded speaker, and genre
   (config in `config/`).
2. **Five segmentation sets** — chapter, rolling 1000-word windows, claimed-narrator and
   by-speaker macro-segments, genre aggregates, and a biblical-quotation-removed variant.
3. **Biblical-quotation detection** via KJV 7-gram shingling (flags the Isaiah blocks in
   2 Nephi, Isaiah 53 in Mosiah 14, the Sermon material in 3 Nephi, etc.).
4. **Stylometric features** — function-word frequencies, char/word/POS n-grams,
   sentence-length stats, TTR/MATTR, hapax/dis, punctuation rates, and Burrows's Delta
   vectors. Produced in **full** and **no-punctuation** variants (1830 punctuation is the
   typesetter's, not authorial).
5. **Method-validation harness** — calibrates the pipeline on KJV where authorship is
   known (OT vs NT; Pauline vs Synoptic vs Johannine) and on null controls (arbitrary
   Isaiah halves; BoM label permutation). This is a **gate**: if positive controls fail,
   the Book of Mormon attribution is marked `UNCALIBRATED`.
6. **Unsupervised analysis** — PCA/UMAP projection, clustering, and cluster agreement
   (ARI/AMI) against narrator/genre/editorial-layer/quote labels; Delta heatmaps.
7. **Supervised attribution** — leave-one-chapter-out cross-validation classifying chapters
   by narrator, run as a 2×2×2 grid (include/exclude quotes × raw/narrative-only ×
   full/nopunct), with a word-count-only confound probe.
8. **Report** — JSON + Markdown with figures, calibration shown before the BoM results.

## Setup

```bash
pip install -r ../requirements.txt
python -m spacy download en_core_web_sm   # one-time spaCy model download
```

No API keys are required for phase one (everything runs on the in-repo source JSON).

## Run

```bash
# full pipeline
python authorship/scripts/run_authorship_pipeline.py

# preview without executing
python authorship/scripts/run_authorship_pipeline.py --dry-run

# fast smoke run (reduced feature vocab)
python authorship/scripts/run_authorship_pipeline.py --quick

# resume from a step, reusing a run id (steps 5-8 share the run id)
python authorship/scripts/run_authorship_pipeline.py --start-from 6 --run-id run_YYYYMMDD_HHMMSS
```

Individual steps can also be run directly (see each script's docstring).

## Layout

```
config/    authorship_config.json, narrator_map.json, embedded_speakers.json, genre_map.json
data/      segments/ (segmentation sets + index), quotes/ (KJV index, spans, fractions), validation/
features/  feature_*.parquet, delta_matrix_*.parquet, feature_manifest.json   (gitignored)
scripts/   the 8 pipeline steps + authorship_common.py + stylometry_metrics.py
results/   raw/ (per-run JSON), figures/ (PNG), reports/ (JSON + Markdown)      (gitignored)
```

## How to read the results

- **Calibration first.** The positive controls bound what the method can detect; the
  negative controls confirm it does not invent authorship in null data.
- **Watch the confounds.** Genre and segment length are entangled with the narrator labels;
  prefer the genre-restricted, quote-excluded, no-punctuation readings and check the
  word-count probe before trusting raw-condition accuracy.
- **H3 vs H4 cannot be settled here.** Internal voice separation is consistent with both a
  multi-voice ancient record and a single translator rendering distinct sources. Phase one
  measures *whether* internal variation exists and survives controls; it does not decide
  the production model, and it says nothing about H1/H2.
