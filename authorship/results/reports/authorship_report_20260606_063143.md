# Book of Mormon Authorship / Production Study — Phase One Report

**Run ID:** run_20260606_063143
**Generated:** 2026-06-06 06:37:04
**Scope:** Internal multi-voice (H3) and translation-layer (H4) analysis using only in-repo texts (Book of Mormon + KJV control). External-author hypotheses (H1/H2) are explicitly out of scope for phase one.

## 1. Corpus & Segmentation

| Property | Value |
| --- | --- |
| Volume | Book of Mormon |
| Books | 15 |
| Chapters | 239 |
| Verses | 6604 |
| Chapter segments | 239 |
| Rolling-1000 windows | 270 |
| Narrator+speaker segments | 19 |
| Genre segments | 5 |


## 2. Narrator Reliability

| Claimed narrator | Words | Reliability |
| --- | --- | --- |
| Mormon | 174014 | reliable |
| Nephi | 54504 | reliable |
| Moroni | 26000 | reliable |
| Jacob | 9135 | reliable |
| Omni et al. | 1398 | composite_unreliable |
| Enos | 1156 | report_only |
| Jarom | 731 | report_only |


Report-only / composite narrators are excluded from supervised attribution.

## 3. Biblical-Quotation Detection (KJV control)

| Chapter | Quote fraction | Top KJV source |
| --- | --- | --- |
| 2 Nephi 22 | 1.00 | Isaiah 12, Exodus 15 |
| 3 Nephi 25 | 1.00 | Malachi 4 |
| 2 Nephi 19 | 1.00 | Isaiah 9, Isaiah 5 |
| 2 Nephi 21 | 0.99 | Isaiah 11, Isaiah 7 |
| Mosiah 14 | 0.99 | Isaiah 53, Psalms 52 |
| 2 Nephi 17 | 0.99 | Isaiah 7, 2 Kings 15 |
| 2 Nephi 18 | 0.99 | Isaiah 8, Genesis 30 |
| 2 Nephi 20 | 0.99 | Isaiah 10, Isaiah 5 |
| 3 Nephi 22 | 0.98 | Isaiah 54, Isaiah 47 |
| 2 Nephi 15 | 0.98 | Isaiah 5, Psalms 106 |


**Known-control check:**
| Chapter | Expected source | Detected fraction | Detected source |
| --- | --- | --- | --- |
| 2 Nephi 22 | Isaiah 12 | 1.00 | Isaiah 12, Exodus 15 |
| 2 Nephi 19 | Isaiah 9 | 1.00 | Isaiah 9, Isaiah 5 |
| Mosiah 14 | Isaiah 53 | 0.99 | Isaiah 53, Psalms 52 |
| 3 Nephi 25 | Malachi 4 | 1.00 | Malachi 4 |
| 3 Nephi 13 | Matthew 6 | 0.90 | Matthew 6, Matthew 13 |


## 4. Method Validation / Calibration

| Control | Type | Result | Gate / chance | Verdict |
| --- | --- | --- | --- | --- |
| OT vs NT | positive | macro-F1 0.990 | ≥ 0.85 | PASS |
| Pauline / Synoptic / Johannine | positive | macro-F1 0.963 | ≥ 0.65 | PASS |
| Isaiah arbitrary halves | negative | acc 0.470 | chance 0.50 | OK |
| BoM label permutation | negative | acc 0.524±0.030 | majority 0.62 | OK |


**Overall calibration: PASSED.** The positive controls show the pipeline separates genuinely distinct authors within a single translation (KJV); the negative controls show it does not invent authorship in null data. This bounds how to read the Book of Mormon results below.

## 5. Unsupervised Structure

Best KMeans: k=3, silhouette=0.125.

Cluster agreement (Adjusted Rand Index) of the best clustering against each label set:

| Label set | ARI | AMI |
| --- | --- | --- |
| claimed_narrator | 0.085 | 0.060 |
| genre | 0.318 | 0.264 |
| editorial_layer | 0.069 | 0.075 |
| quote_status | 0.216 | 0.344 |


> **Headline caveat:** clusters agree with **genre** more than with **narrator** — much apparent voice separation is structural (sermon vs narrative vs prophecy).

Figures: `figures/run_20260606_063143/umap_{claimed_narrator,genre,editorial_layer,quote_status}.png`

## 6. Burrows's Delta Between Narrators

| Δ | Jacob | Mormon | Moroni | Nephi |
| --- | --- | --- | --- | --- |
| Jacob | 0.00 | 0.41 | 0.41 | 0.37 |
| Mormon | 0.41 | 0.00 | 0.32 | 0.35 |
| Moroni | 0.41 | 0.32 | 0.00 | 0.31 |
| Nephi | 0.37 | 0.35 | 0.31 | 0.00 |


Heatmaps (full and quote-removed): `figures/run_20260606_063143/delta_narrators_full.png`, `figures/run_20260606_063143/delta_narrators_quote_removed.png`

## 7. Supervised Attribution (leave-one-chapter-out)

| Condition (quote|genre|features) | Classifier | n | classes | Accuracy | Macro-F1 | Baseline |
| --- | --- | --- | --- | --- | --- | --- |
| include_quotes|raw|full | logistic | 236 | 4 | 0.898 | 0.640 | 0.114 |
| include_quotes|raw|full | svc_linear | 236 | 4 | 0.877 | 0.668 | 0.114 |
| include_quotes|narrative_only|full | logistic | 144 | 3 | 0.958 | 0.908 | 0.750 |
| include_quotes|narrative_only|full | svc_linear | 144 | 3 | 0.868 | 0.792 | 0.750 |
| exclude_quotes|raw|full | logistic | 213 | 4 | 0.897 | 0.634 | 0.127 |
| exclude_quotes|raw|full | svc_linear | 213 | 4 | 0.859 | 0.618 | 0.127 |
| exclude_quotes|narrative_only|full | logistic | 142 | 3 | 0.958 | 0.911 | 0.761 |
| exclude_quotes|narrative_only|full | svc_linear | 142 | 3 | 0.873 | 0.799 | 0.761 |
| include_quotes|raw|nopunct | logistic | 236 | 4 | 0.890 | 0.628 | 0.114 |
| include_quotes|raw|nopunct | svc_linear | 236 | 4 | 0.877 | 0.673 | 0.114 |
| include_quotes|narrative_only|nopunct | logistic | 144 | 3 | 0.958 | 0.908 | 0.750 |
| include_quotes|narrative_only|nopunct | svc_linear | 144 | 3 | 0.889 | 0.827 | 0.750 |
| exclude_quotes|raw|nopunct | logistic | 213 | 4 | 0.897 | 0.677 | 0.127 |
| exclude_quotes|raw|nopunct | svc_linear | 213 | 4 | 0.840 | 0.592 | 0.127 |
| exclude_quotes|narrative_only|nopunct | logistic | 142 | 3 | 0.965 | 0.928 | 0.761 |
| exclude_quotes|narrative_only|nopunct | svc_linear | 142 | 3 | 0.880 | 0.812 | 0.761 |


Confusion matrix (best raw cell): `figures/run_20260606_063143/confusion_best.png`

## 8. Confound Controls

- **Length probe:** narrator predicted from word-count alone = **0.623** accuracy. A high value here means length is entangled with the narrator labels and partially inflates attribution; treat raw-condition accuracy with caution and prefer the genre-restricted, length-matched readings.
- **Quotation:** every cell is reported include vs exclude quotes; persistence of the signal across both indicates it is not merely shared KJV diction.
- **Genre:** narrative-only cells isolate the genre confound; persistence there indicates residual narrator (or length/topic) signal beyond genre.
- **Punctuation:** full vs nopunct variants; persistence rules out the 1830 typesetter's punctuation as the driver.

## 9. Caveats & Interpretation

- **Translation-layer collapse (H3 vs H4):** all Book of Mormon text shares one 1829 dictation idiom; internal separation is consistent with BOTH a multi-voice ancient record AND a single translator rendering distinct sources. Calibration shows the method *could* detect distinct authors if present, but cannot by itself decide between H3 and H4.
- **Editorial homogenization:** Mormon/Moroni abridged most books, so 'narrator' classes largely reflect the abridging editor; embedded-speaker voices are 'as preserved in the abridgement,' not necessarily original diction.
- **Length confound:** see §8; word-count entanglement is the main internal threat to the raw-condition results.
- **Archaic-text NLP:** spaCy POS/sentence segmentation is approximate on KJV-style English; function-word and char-n-gram features (more robust) carry the headline claims.
- **Phase-one scope:** this analysis cannot address whether Joseph Smith or any 19th-century author wrote the text (H1/H2). That requires the external corpus (phase two).