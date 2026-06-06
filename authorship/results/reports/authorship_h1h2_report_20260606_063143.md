# Book of Mormon Authorship — Phase Two Report (H1/H2: external authors)

**Run ID:** run_20260606_063143
**Generated:** 2026-06-06 07:01:11
**Question:** Does the English text of the Book of Mormon stylometrically resemble any candidate 19th-century author, with an explicit *none of the above* option?

## Read this first: the register confound

Among the comparison authors, only **Joseph Smith (Doctrine and Covenants)** and the **KJV** share the Book of Mormon's dictated / scriptural-archaic register. The source-theory candidates (Ethan Smith, Solomon Spalding) and the distractors (Edwards, Bunyan, Irving) are ordinary written prose. Any pull toward Joseph Smith or the KJV therefore partly reflects **register**, not necessarily authorship. A cleaner test would require same-register comparators that do not exist for the source-theory candidates. Read every result below through this caveat.

## 1. Comparison corpus

| Author | Role | Register | Windows | OCR noise | Source |
| --- | --- | --- | --- | --- | --- |
| Joseph Smith | candidate | dictated_revelation | 45 | n/a | in-repo scriptures/source/doctrine-and-covenants |
| Ethan Smith | candidate | written_treatise | 45 | 10% | archive.org/details/viewofhebrewsort00smit — Vie |
| Solomon Spalding | candidate | written_narrative | 41 | 6% | archive.org/details/manuscriptfoundm00spau — 'Th |
| Jonathan Edwards | distractor | sermon | 45 | 4% | gutenberg.org/ebooks/34632 — Selected Sermons of |
| John Bunyan | distractor | religious_allegory | 45 | 6% | gutenberg.org/ebooks/131 — The Pilgrim's Progres |
| Washington Irving | distractor | literary_prose | 45 | 4% | gutenberg.org/ebooks/2048 — The Sketch-Book of G |
| KJV Bible (translation control) | translation_control | early_modern_translation | 45 | n/a | in-repo scriptures/source/new-testament.json (KJ |


- **Gap — Sidney Rigdon:** No clean public-domain prose corpus readily fetchable from Gutenberg/archive.org full-text endpoints; excluded rather than approximated.

## 2. Method sanity

- **Closed-set author accuracy:** 0.977 across 7 authors — the candidate/distractor authors are strongly separable from each other, so the feature space carries real authorial signal.
- **Open-set calibration (leave-one-author-out):** at threshold 0.992, a genuinely unseen author is rejected 90% of the time while known authors are retained 61% of the time. (Unknown median max-prob 0.78 vs known 1.00.)

## 3. Book of Mormon attribution — probabilistic (logistic + rejection)

- **73% of Book of Mormon windows are rejected as _none of the above_** (196 of 270).
- Among the windows that are NOT rejected, the predicted author is:
| Author | Windows assigned |
| --- | --- |
| Joseph Smith | 74 |


Mean per-author probability over all BoM windows:

| Author | Mean P(author | BoM window) |
| --- | --- |
| Joseph Smith | 0.849 |
| KJV Bible (translation control) | 0.123 |
| John Bunyan | 0.023 |
| Jonathan Edwards | 0.002 |
| Ethan Smith | 0.001 |
| Solomon Spalding | 0.001 |
| Washington Irving | 0.000 |


Figure: `figures/run_20260606_063143/openset_mean_probability.png`

## 4. Book of Mormon attribution — distance (Burrows's Delta)

Nearest author by window (count of BoM windows whose nearest author centroid is X):

| Author | BoM windows nearest |
| --- | --- |
| Joseph Smith | 107 |
| KJV Bible (translation control) | 47 |
| Ethan Smith | 45 |
| Washington Irving | 39 |
| Solomon Spalding | 23 |
| John Bunyan | 5 |
| Jonathan Edwards | 4 |


Delta distance from the BoM centroid to each author (lower = closer):

| Author | Delta distance |
| --- | --- |
| KJV Bible (translation control) | 0.451 |
| Joseph Smith | 0.451 |
| John Bunyan | 0.560 |
| Ethan Smith | 0.576 |
| Jonathan Edwards | 0.579 |
| Solomon Spalding | 0.611 |
| Washington Irving | 0.628 |


**Nearest overall (Delta centroid): KJV Bible (translation control).** Figure: `figures/run_20260606_063143/openset_delta_distance.png`

## 5. Interpretation

- The Book of Mormon does **not** cleanly match any single candidate author: the open-set classifier rejects 73% of its windows as none-of-the-above.
- Where it is pulled toward a candidate, it is pulled toward **Joseph Smith's dictated revelations (D&C)** and the **KJV** — the two texts sharing its register — and **away from** the written-prose source-theory candidates (Ethan Smith, Solomon Spalding) and the distractors.
- On the tested evidence, the source-text hypotheses (H2: Spalding / Ethan Smith authorship of the English text) are **not** supported: the Book of Mormon is stylometrically far from those authors. The data are more consistent with the Joseph-Smith-register account (H1) than with the source-author account — **but** this is confounded by register and cannot, alone, distinguish single authorship from a translator rendering the text in a KJV-like register (H4).

## 6. Limitations

- **Register confound (dominant):** see the top of this report.
- **Joseph Smith proxy:** the only public-domain Joseph Smith sample used is the D&C, which is itself dictated revelation — it shares the BoM's production mode, so resemblance to it is expected under several hypotheses.
- **OCR noise:** Ethan Smith and Spalding come from archive.org OCR (see noise column); function-word/char features are robust but some signal is degraded.
- **Corpus gap:** no clean public-domain Sidney Rigdon prose corpus was obtained; the Rigdon variant of H2 is untested here.
- **Small candidate set & window count:** a handful of authors and ~45 windows each; broader corpora would tighten the open-set boundary.
- **English text only:** stylometry speaks to the English text's production; it cannot address divine translation, an inherently metaphysical claim.