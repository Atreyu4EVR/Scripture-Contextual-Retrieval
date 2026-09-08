# Faith Questions Dataset

## Mission

Build a reproducible pipeline that collects publicly available questions, objections, and doubt statements about Latter-day Saint doctrine, history, and practice, normalizes them into a labeled corpus, and packages the result as a Hugging Face dataset for downstream ML work (evaluation sets, intent routing, retrieval benchmarks).

The output is a dataset, not an answering system. Nothing in this repo generates responses to the questions it collects.

## Non-Negotiable Rules

These are gates, not guidelines. Stop and ask the human rather than working around any of them.

1. **No fetcher without a resolved license.** A source gets a fetcher only after `sources/<name>.yaml` exists with `license`, `permission_status`, and `redistributable` populated by a human. Unresolved means stop
2. **robots.txt is parsed, not eyeballed.** Use `protego`. Honor `Crawl-delay`. If a path is disallowed, it is not fetched
3. **Identify yourself.** Every request carries a descriptive User-Agent including a project name and a contact email address
4. **Rate limit by default.** One request per two seconds per domain, backing off on any 429 or 5xx. Configurable upward only by the human
5. **Raw data never enters git.** `data/` is gitignored in full. Raw archives live on local disk or Azure Blob
6. **Never push public.** `huggingface_hub` pushes are private-only in code. Visibility is flipped by a human in the HF UI, never by a script
7. **Verbatim text is conditional.** A record carries `verbatim_text` only when its source has `redistributable: true` and `author_attribution` is populated. Otherwise the record carries the normalized question form alone
8. **PII scrubbing precedes storage.** No usernames, real names, email addresses, ward or stake identifiers, or specific locations survive into `data/parsed/`. The single exception is `author_attribution`, which carries only the credit CC BY-SA legally requires (author name and links); scrubbing applies to every other text field

## Content Sensitivity

This corpus consists of real people describing loss of faith, family rupture, and religious doubt. Treat it accordingly:

- Never construct records that link a question back to an identifiable individual
- Never build features that infer religious status, mental health, or sexual orientation about a source author
- Preserve the question, discard the biography

## Repository Layout

```
taxonomy/
  issues.yaml           # human-authored issue taxonomy, the spine of the project
  registers.yaml        # question register / stance vocabulary
sources/
  <source>.yaml         # one manifest per source: license, permission, endpoints, limits
  permissions/          # scanned or saved permission correspondence
src/faithqs/
  schema.py             # pydantic v2 record models
  fetch/                # per-source fetchers, one module per source
  parse/                # per-source parsers, one module per source
  scrub.py              # PII removal
  extract.py            # question-form extraction
  classify.py           # taxonomy assignment
  review/               # human review CLI
  package.py            # HF dataset build and push
data/                   # gitignored
  raw/                  # immutable, content-hashed, never edited in place
  staged/               # parse output, unscrubbed, never leaves local disk
  parsed/               # scrubbed only; nothing enters without pii_scrubbed: true
  reviewed/
  release/
tests/
```

## Record Schema

Define in `src/faithqs/schema.py` with pydantic v2. Every field below is required unless marked nullable. Stages before `classify` cannot populate the taxonomy fields, so `parse` emits the source-side subset as a `SourceRecord` (source identity, license, attribution, raw title and body text, tags); `classify` is the first stage that emits a complete record.

| Field | Type | Notes |
|---|---|---|
| `record_id` | str | UUID, stable across runs: v5 derived from `source_name` and `source_record_id` for collected records, v4 for phrasing variants we author |
| `question_text` | str | Normalized canonical question, our work product |
| `verbatim_text` | str \| None | Populated only when redistribution is permitted |
| `issue_id` | str | Foreign key into `taxonomy/issues.yaml` |
| `issue_category` | str | Denormalized parent category |
| `register` | str | From `taxonomy/registers.yaml`, human-reviewed |
| `classifier_confidence` | float \| None | 0 to 1, emitted by `classify` and used to prioritize review; null before classification |
| `source_name` | str | Matches a file in `sources/` |
| `source_url` | str \| None | Null where the source is a bulk dump |
| `source_record_id` | str | Native identifier in the source system |
| `source_license` | str | SPDX identifier or explicit enum value |
| `author_attribution` | str \| None | Required when the license is CC BY-SA |
| `collected_at` | datetime | UTC |
| `redistributable` | bool | Drives split assignment |
| `permission_ref` | str \| None | Path under `sources/permissions/` |
| `pii_scrubbed` | bool | Must be true before a record enters `data/parsed/` |
| `review_status` | enum | `auto`, `human_reviewed`, `quarantined` |

Any record failing validation goes to `data/parsed/quarantine/` with the failure reason attached. Quarantined records never reach a release split.

## Two-Tier Output

The release is split by redistribution rights, decided per source, not per record type.

**Tier A, publishable.** Sources under CC BY-SA 4.0 or equivalent, plus our own taxonomy labels and any synthetic phrasing variants we author. Note that ShareAlike propagates: a Tier A release containing CC BY-SA text must itself carry CC BY-SA 4.0 and per-record attribution. Design the dataset card around that obligation rather than discovering it at push time.

**Tier B, private.** Everything else. Content under restrictive terms, commercial keyword exports, copyrighted published works, and any source whose permission scope covers collection but not redistribution. Where feasible, Tier B stores identifiers and our derived labels rather than source text, so the artifact is a pointer set rather than a copy.

## Pipeline Stages

Each stage is independently runnable and idempotent, reading from the prior stage's output directory.

1. `fetch` writes immutable content-hashed payloads to `data/raw/<source>/`
2. `parse` runs the source adapter, producing `SourceRecord` JSONL in `data/staged/<source>/`, named by the raw payload hash
3. `scrub` strips PII from every text field except `author_attribution`, sets `pii_scrubbed`, and is the only stage that writes `data/parsed/<source>/`
4. `extract` derives the normalized question form from source text, with LLM calls cached on content hash
5. `classify` assigns `issue_id` and `register` against the taxonomy, emitting confidence
6. `review` presents low-confidence and sampled-high-confidence records in a CLI for human adjudication
7. `package` builds the HF `DatasetDict` and pushes private

Never chain stages into a single command that hides intermediate state. Re-running `extract` must not require re-fetching.

## Taxonomy First

`taxonomy/issues.yaml` is human-authored before any fetcher is written, seeded from the concern lists that already exist publicly (CES Letter section structure, Gospel Topics Essays, Mormonr Q&A categories, FAIR's answer index). Those documents inform structure only and are not ingested as data.

Expect roughly forty to sixty top-level issues. If classification produces a large residual bucket, the taxonomy is wrong and needs revision before more collection happens.

## Source Manifest Format

```yaml
name: christianity-stackexchange
kind: bulk_dump
license: CC-BY-SA-4.0
license_url: https://creativecommons.org/licenses/by-sa/4.0/
redistributable: true
attribution_required: true
permission_status: not_required
notes: >
  Internet Archive releases through 2024-04 only. Later dumps carry
  restrictions incompatible with this use. Filter to LDS and Mormonism tags.
endpoints:
  - https://archive.org/details/stackexchange
rate_limit_seconds: 2
```

## Stack

Python 3.12 managed with `uv`. `httpx` for transport, `protego` for robots parsing, `selectolax` for HTML, `pydantic` v2 for schema, `duckdb` for intermediate querying, `datasets` and `huggingface_hub` for packaging, `pytest` for tests. Type hints throughout, `ruff` for lint and format.

## Build Order

Do not skip ahead. Each milestone ends with a human checkpoint.

- **M0** Taxonomy, schema, and one bulk-dump source that requires no crawling. Proves the record model end to end
- **M1** Scrub, extract, classify, and the review CLI running over M0 output
- **M2** A second source through the same adapter interface, proving the abstraction
- **M3** A permission-gated forum adapter, only after written permission is filed under `sources/permissions/`
- **M4** Dataset card and private HF push

## Dataset Card Requirements

The card is part of the deliverable, not an afterthought. It must state: per-split license, a provenance table naming every source with its terms, intended uses, explicitly out-of-scope uses, the PII handling method, and a plain statement that the corpus reflects real expressions of religious doubt by identifiable communities.

## Escalate, Do Not Improvise

Stop and ask the human when you encounter any of the following:

- A source whose terms are ambiguous on redistribution or ML use
- A robots.txt that blocks the content we want
- A parser that would need to defeat rate limiting, bot detection, or a login wall
- A classification residual bucket exceeding fifteen percent
- Any request to make a dataset public

## Amendment Log

- 2026-09-07: M0 checkpoint. The project owner approved the taxonomy and register drafts. Rule 8 gained the `author_attribution` exception; `data/staged/` was added so unscrubbed parse output never touches `data/parsed/`; `classifier_confidence` joined the record schema; `record_id` became a source-derived UUIDv5 so ids are stable across runs; `SourceRecord` was named as the `parse` output.
