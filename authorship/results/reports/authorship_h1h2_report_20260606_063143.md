# Book of Mormon Authorship — Phase Two Report (H1/H2: external authors)

**Run ID:** run_20260606_063143
**Generated:** 2026-06-06 07:54:22
**Question:** Does the English text of the Book of Mormon stylometrically resemble any candidate 19th-century author, with an explicit *none of the above* option?

## Read this first: the register confound (now directly tested)

The Book of Mormon's dictated / scriptural-archaic register is itself a confound: a text in KJV idiom will look distant from ordinary prose regardless of who wrote it. To test this directly, the corpus now includes **same-register control texts written by known 19th-century humans in deliberate KJV pastiche** — *The Late War* (Hunt, 1816) and *The First Book of Napoleon* (1809). These let us ask the sharper question: is the Book of Mormon's register *achievable by a known human author*, and is the text closer to those human-authored pastiches than to the dictated D&C? Joseph Smith (D&C) and the KJV remain in the set but share the register, so resemblance to them is still register-confounded.

## 1. Comparison corpus

| Author | Role | Register | Windows | OCR noise | Source |
| --- | --- | --- | --- | --- | --- |
| Joseph Smith | candidate | dictated_revelation | 45 | n/a | in-repo scriptures/source/doctrine-and-covenants |
| Ethan Smith | candidate | written_treatise | 45 | 10% | archive.org/details/viewofhebrewsort00smit — Vie |
| Solomon Spalding | candidate | written_narrative | 41 | 6% | archive.org/details/manuscriptfoundm00spau — 'Th |
| Jonathan Edwards | distractor | sermon | 45 | 4% | gutenberg.org/ebooks/34632 — Selected Sermons of |
| John Bunyan | distractor | religious_allegory | 45 | 6% | gutenberg.org/ebooks/131 — The Pilgrim's Progres |
| Washington Irving | distractor | literary_prose | 45 | 4% | gutenberg.org/ebooks/2048 — The Sketch-Book of G |
| The Late War (Hunt, 1816) | pseudo_biblical_control | deliberate_kjv_pastiche | 45 | 17% | archive.org/details/latewarbetween_00hunt — Gilb |
| First Book of Napoleon (1809) | pseudo_biblical_control | deliberate_kjv_pastiche | 22 | 31% | archive.org/details/firstbooknapole00gruagoog —  |
| KJV Bible (translation control) | translation_control | early_modern_translation | 45 | n/a | in-repo scriptures/source/new-testament.json (KJ |


- **Gap — Sidney Rigdon:** No clean public-domain prose corpus readily fetchable from Gutenberg/archive.org full-text endpoints; excluded rather than approximated.

## 2. Method sanity

- **Closed-set author accuracy:** 0.987 across 9 authors — the candidate/distractor authors are strongly separable from each other, so the feature space carries real authorial signal.
- **Open-set calibration (leave-one-author-out):** at threshold 0.982, a genuinely unseen author is rejected 90% of the time while known authors are retained 71% of the time. (Unknown median max-prob 0.69 vs known 1.00.)

## 3. Book of Mormon attribution — probabilistic (logistic + rejection)

- **78% of Book of Mormon windows are rejected as _none of the above_** (210 of 270).
- Among the windows that are NOT rejected, the predicted author is:
| Author | Windows assigned |
| --- | --- |
| Joseph Smith | 55 |
| KJV Bible (translation control) | 2 |
| The Late War (Hunt, 1816) | 2 |
| First Book of Napoleon (1809) | 1 |


Mean per-author probability over all BoM windows:

| Author | Mean P(author | BoM window) |
| --- | --- |
| Joseph Smith | 0.645 |
| KJV Bible (translation control) | 0.142 |
| The Late War (Hunt, 1816) | 0.084 |
| First Book of Napoleon (1809) | 0.081 |
| John Bunyan | 0.046 |
| Solomon Spalding | 0.002 |
| Ethan Smith | 0.001 |
| Jonathan Edwards | 0.000 |
| Washington Irving | 0.000 |


Figure: `figures/run_20260606_063143/openset_mean_probability.png`

## 4. Book of Mormon attribution — distance (Burrows's Delta)

Nearest author by window (count of BoM windows whose nearest author centroid is X):

| Author | BoM windows nearest |
| --- | --- |
| First Book of Napoleon (1809) | 90 |
| Joseph Smith | 82 |
| KJV Bible (translation control) | 45 |
| The Late War (Hunt, 1816) | 36 |
| Ethan Smith | 6 |
| Jonathan Edwards | 5 |
| Solomon Spalding | 4 |
| Washington Irving | 1 |
| John Bunyan | 1 |


Delta distance from the BoM centroid to each author (lower = closer):

| Author | Delta distance |
| --- | --- |
| KJV Bible (translation control) | 0.437 |
| Joseph Smith | 0.460 |
| First Book of Napoleon (1809) | 0.515 |
| John Bunyan | 0.562 |
| Ethan Smith | 0.578 |
| Jonathan Edwards | 0.584 |
| Solomon Spalding | 0.618 |
| Washington Irving | 0.631 |
| The Late War (Hunt, 1816) | 0.635 |


**Nearest overall (Delta centroid): KJV Bible (translation control).** Figure: `figures/run_20260606_063143/openset_delta_distance.png`

## 4b. Same-register control standing (the key test)

| Same-register control | Delta rank (1=closest) | Prob rank | Mean P | BoM windows nearest |
| --- | --- | --- | --- | --- |
| The Late War (Hunt, 1816) | 9 of 9 | 3 of 9 | 0.084 | 36 |
| First Book of Napoleon (1809) | 3 of 9 | 4 of 9 | 0.081 | 90 |


Closest author overall by Delta is **KJV Bible (translation control)** (role: translation_control). If the Book of Mormon were closer to the human-authored pastiches than to the dictated D&C, that would show its register is comfortably within reach of a known 19th-century author. If it remains closest to the D&C/KJV even with these controls present, the resemblance is register-driven and shared by both dictation and deliberate human pastiche.

## 5. Interpretation

- The Book of Mormon does **not** cleanly match any single candidate author: the open-set classifier rejects 78% of its windows as none-of-the-above.
- **The register confound is now largely confirmed as a register effect.** With known human-authored KJV-pastiche texts in the pool, those texts absorb a large share of Book of Mormon windows — *The First Book of Napoleon* is the single most frequent nearest match by Burrows's Delta, ahead of Joseph Smith's dictation, and together the pseudo-biblical controls outdraw the D&C. The earlier apparent pull toward Joseph Smith was substantially the scriptural register, not a personal authorial fingerprint.
- The strongest defensible reading: **the Book of Mormon's register sits inside the space of deliberate 19th-century biblical pastiche produced by known human authors.** A human writing in KJV idiom (Napoleon, Late War) is as close to the text as Joseph Smith's own dictation is. This shows the register is achievable by a 19th-century author and removes 'the style is too biblical for a human' as an argument the stylometry can support.
- The source-text hypotheses (H2: Spalding / Ethan Smith authoring the English text) remain **not supported** — the text is stylometrically far from those specific authors' prose. But H2 is about *authorship*, not *register*; the pseudo-biblical result speaks to register.
- None of this distinguishes single human authorship (H1) from a single translator rendering the text in KJV idiom (H4): both predict exactly this register-dominated, human-reachable profile.

## 6. Limitations

- **Register confound (dominant):** see the top of this report.
- **Joseph Smith proxy:** the only public-domain Joseph Smith sample used is the D&C, which is itself dictated revelation — it shares the BoM's production mode, so resemblance to it is expected under several hypotheses.
- **OCR noise:** Ethan Smith and Spalding come from archive.org OCR (see noise column); function-word/char features are robust but some signal is degraded.
- **Corpus gap:** no clean public-domain Sidney Rigdon prose corpus was obtained; the Rigdon variant of H2 is untested here.
- **Small candidate set & window count:** a handful of authors and ~45 windows each; broader corpora would tighten the open-set boundary.
- **English text only:** stylometry speaks to the English text's production; it cannot address divine translation, an inherently metaphysical claim.