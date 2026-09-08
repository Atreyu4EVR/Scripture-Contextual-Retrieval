# Faith Questions Dataset

A reproducible pipeline that collects publicly available questions, objections, and doubt statements about Latter-day Saint doctrine, history, and practice, normalizes them into a labeled corpus, and packages the result as a private Hugging Face dataset. The output is a dataset, not an answering system. Governance, gates, and build order live in [CLAUDE.md](CLAUDE.md), which is the authority wherever this README and that file disagree.

## Status

| Milestone | State |
|---|---|
| M0: taxonomy, schema, one bulk-dump source | Code complete. Taxonomy approved 2026-09-07. Fetcher and parser for `christianity-stackexchange` in place with an offline end-to-end proof. Awaiting the owner's first live run (data never enters git, so it happens on a local machine). |
| M1: scrub, extract, classify, review CLI | Not started |
| M2: second source | Not started |
| M3: permission-gated forum adapter | Not started |
| M4: dataset card and private HF push | Not started |

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
cd faith-questions-dataset
uv sync
uv run pytest
uv run ruff check
```

Every network request identifies the project and a contact address (CLAUDE.md rule 3), and the address is yours to supply:

```bash
export FAITHQS_CONTACT_EMAIL=you@example.org   # required before fetch
export FAITHQS_DATA_DIR=./data                 # optional, this is the default

uv run faithqs fetch christianity-stackexchange   # data/raw/<source>/<sha256>.7z + .meta.json
uv run faithqs parse christianity-stackexchange   # data/staged/<source>/<sha256>.jsonl + reports
```

Stages are separate commands on purpose. Re-running `parse` never re-fetches; re-running `fetch` on an unchanged upstream costs one HEAD request.

The parse step also writes `<sha256>.report.json` with tag frequencies across the whole dump and within the selection. Use it to tune `filters.tags` in `sources/christianity-stackexchange.yaml`, then re-run `parse --force`. The canonical tag on Christianity Stack Exchange is `lds` (597 questions in the archived dump); `mormonism` and `latter-day-saints` are synonyms that never appear as stored tags.

The archived dump is frozen: Stack Exchange stopped uploading to archive.org after the 2024-04 release, so the manifest pins the file's published sha1, md5, and size and the fetcher discards any download that does not match.

## Source terms flagged for owner review

The Internet Archive's Terms of Use grant access "for scholarship and research purposes only" and ask users "not to collect or store personal data about anyone." This project reads only `DisplayName` from `Users.xml`, for the attribution CC BY-SA requires, and never loads location, bio, or website fields. Whether that satisfies the Archive's terms is a judgment for the project owner, recorded in the manifest notes. Stack Exchange's own post-2024 distribution channel adds terms on model training that the archived 2024-04 file does not carry.

## What stops a run

Each gate in CLAUDE.md is enforced in code and surfaces as a `STOP:` line with exit code 2:

- Taxonomy or register file with `approved: false` (Taxonomy First)
- Manifest with an unresolved license or permission, or a granted permission with no filed correspondence (rule 1)
- robots.txt disallowing the path, or unreachable (rule 2)
- Missing `FAITHQS_CONTACT_EMAIL` (rule 3)
- Any attempt to set a rate limit under two seconds (rule 4)
- A record reaching `data/parsed/` with `pii_scrubbed: false` (rule 8)

## Layout

```
CLAUDE.md                 # project constitution: rules, schema, build order, amendment log
taxonomy/
  issues.yaml             # approved issue taxonomy (60 issues, 13 categories)
  registers.yaml          # approved register vocabulary (8 registers)
sources/
  christianity-stackexchange.yaml   # M0 source manifest, human-resolved license
  permissions/            # permission correspondence (referenced by permission_ref)
src/faithqs/
  schema.py               # SourceRecord, FaithQuestionRecord, quarantine, license and PII gates
  taxonomy.py             # taxonomy loaders, approval gate, foreign-key checks
  manifest.py             # source manifests and the rule 1 gate
  config.py               # settings from the environment, User-Agent construction
  storage.py              # content-addressed data/raw and data/staged layout
  cli.py                  # `faithqs fetch|parse <source>`
  fetch/polite.py         # httpx client enforcing robots.txt, User-Agent, rate limit, backoff
  fetch/bulk_dump.py      # generic single-file downloader: ETag skip, pinned checksum verification
  fetch/christianity_stackexchange.py
  parse/stackexchange.py  # 7z -> Posts.xml/Users.xml -> SourceRecord JSONL, license per post,
                          #   both tag encodings, watermark and deleted rows skipped
  parse/christianity_stackexchange.py
tests/                    # 90+ offline tests, including a fetch -> parse -> record proof
data/                     # gitignored in full; created at runtime
```

## Licensing notes for the dataset card

Stack Exchange content is CC BY-SA at the version in force when each post, or its latest revision, was made (2.5, then 3.0 from 2011-04-08, then 4.0 from 2018-05-02). The parser records the exact version per record from the dump's `ContentLicense` attribute, with a creation-date fallback for dumps that lack it. A Tier A release will therefore contain a mix of CC BY-SA 3.0 and 4.0 text. Each record's `author_attribution` carries what Stack Exchange's license terms require: the author's name, a direct link to the author's profile, the originating site named visibly, and a direct link to the original question. The card must state this per-record licensing rather than a single headline version.

## Relationship to Scripture-Contextual-Retrieval

This project is independent of the contextual retrieval work and lives in this repository only because it was scaffolded from a session scoped to it. The directory is self-contained. To extract it into its own repository with history:

```bash
git subtree split --prefix=faith-questions-dataset -b faith-questions-export
# then push that branch to a new repo and remove the directory here
```

A plain copy of the directory also works, since nothing here imports from or links to the parent project.
