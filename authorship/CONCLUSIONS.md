# Conclusions — Book of Mormon Authorship / Production Study

This document synthesizes the phase-one (internal voice, H3/H4) and phase-two (external
authors, H1/H2) results into a single careful conclusion. It follows the scientific-method
framing the project was designed around: state what the data support, what they do not
support, what remains untested, the assumptions that shape interpretation, and how robust
the findings are.

## Research question

> Did Joseph Smith write the Book of Mormon himself, or translate an ancient record?

Testable reframing actually evaluated:

> Which natural-language production model best explains the observable textual features of
> the Book of Mormon's English text?

## Verdict on each hypothesis

**H1 — Joseph Smith sole-author.** *Neither confirmed nor refuted.* The text behaves as one
unified voice and, among the tested candidates, is pulled toward Joseph Smith's own dictated
revelations (D&C) and the KJV. But this is consistent with H1 and H4 equally, and is
confounded by register/production mode. The data do not establish sole authorship.

**H2 — 19th-century source / collaboration.** *Weak-to-moderate lean against, for the
specific authors tested.* The English text is stylometrically far from Solomon Spalding and
Ethan Smith; ~78% of Book of Mormon windows are rejected as "none of the above," and it leans
away from these prose authors. Caveats: part of this distance is just the scriptural register
(any ordinary-prose author would be distant); it cannot refute a heavily-rewritten/re-dictated
source; and Sidney Rigdon was untested (no clean public-domain corpus). So H2 is *not
supported* on the tested evidence, with qualifications.

**Register confound — directly tested (added control).** Same-register, known-human-authored
KJV-pastiche texts (*The Late War*, 1816; *The First Book of Napoleon*, 1809) were added to
the comparison pool. They absorb a large share of Book of Mormon windows — *The First Book of
Napoleon* is the single most frequent nearest match by Burrows's Delta, ahead of Joseph
Smith's dictation. This confirms the Book of Mormon's register sits **inside the space of
deliberate human biblical pastiche**: a known 19th-century human author is stylometrically as
close to the text as Joseph Smith's own dictation. The earlier apparent pull toward Joseph
Smith was substantially a *register* effect, and "the style is too biblical for a human author"
is not a claim the stylometry supports.

**H3 — internal multi-voice.** *Partially present, but largely explained by confounds.*
Chapters classify by claimed narrator at ~0.90 accuracy, and the signal survives removing
biblical quotations and dropping punctuation. However, unsupervised clusters track **genre**
more than **narrator** (ARI 0.32 vs 0.09), and word count alone predicts the narrator at
0.62. Real internal variation exists, but it is not cleanly attributable to distinct
authorial voices once genre, length, and the editorial (Mormon/Moroni abridgement) layer are
accounted for.

**H4 — translation-mediated text.** *Consistent with the data, and not distinguishable from
H1.* A single unifying register plus residual internal variation is exactly what H4 predicts
— but "consistent with" is not "evidence for," and the same pattern is predicted by single
authorship.

## Answer to the core question

**Scientifically, the authored-vs-translated question is undecidable by this method.** This
is a structural limit, not a shortage of data or effort: one person producing one long text
through one process yields a single unified register *whether composing or translating*. H1
and H4 make the same observable prediction, so no stylometric signal can separate them. More
texts, richer features, or larger models do not change this.

The method itself is sound — it was calibrated and **passed**: it separates genuinely
distinct authors within a single translation register (Pauline vs. Synoptic vs. Johannine,
macro-F1 0.96) and does not invent authorship in null data. So the inability to decide the
core question reflects the nature of the question, not a weak pipeline.

## What the evidence does and does not support

Supported:
- The Book of Mormon's English text presents as **one unified register**.
- It does **not** resemble the specific 19th-century source-author candidates tested
  (Spalding, Ethan Smith).
- **Internal stylistic variation exists**, but is largely explained by genre, segment
  length, and the abridging-editor layer.

Not supported (in either direction):
- That Joseph Smith *composed* the text, or that he *translated* it.
- That internal "voices" demonstrate distinct ancient authors.
- Any numeric probability about whether a person of limited formal schooling could have
  authored it — that is a historical/qualitative question outside stylometry's scope, and
  this study neither measured nor can measure it.

Untested / out of scope:
- Sidney Rigdon (corpus gap); same-register external comparators (none exist for the
  source-author candidates); the education-vs-complexity argument; and — by definition — any
  claim of divine translation, which is metaphysical and beyond empirical method.

## Assumptions that shape interpretation

Register confound (dominant); Doctrine and Covenants used as the Joseph Smith proxy (itself
dictated revelation); archive.org OCR noise on Ethan Smith and Spalding; modern-English NLP
applied to archaic text; and coarse rule-based narrator/genre annotation.

## Robustness

Findings are stable across biblical-quote inclusion/exclusion, with and without punctuation,
across two classifier families, and across fixed random seeds. Calibration gates passed.

## Final framing

This study evaluated which production model best accounts for observable patterns in the
Book of Mormon's English text. It does not — and cannot — prove or disprove divine
translation. The most defensible statement the evidence allows:

> The observable textual evidence weakens the 19th-century source-author hypotheses and is
> consistent with a single unifying production register; it cannot distinguish single human
> authorship from single-translator rendering, and it does not address supernatural
> causation.

These findings should be weighed alongside historical, theological, and personal forms of
evidence. They are not decisive for or against any religious truth claim, and should not be
represented as such — in either direction.
