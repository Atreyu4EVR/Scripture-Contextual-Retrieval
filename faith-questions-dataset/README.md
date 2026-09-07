# Faith Questions Dataset

A reproducible pipeline that collects publicly available questions, objections, and doubt statements about Latter-day Saint doctrine, history, and practice, normalizes them into a labeled corpus, and packages the result as a private Hugging Face dataset. The output is a dataset, not an answering system. Governance, gates, and build order live in [CLAUDE.md](CLAUDE.md), which is the authority wherever this README and that file disagree.

## Status

M0 is partially complete and paused at its human checkpoint.

| Milestone | State |
|---|---|
| M0: taxonomy, schema, one bulk-dump source | Schema and tests done. Taxonomy drafted, awaiting human approval. Source manifest filed. Fetcher intentionally not written yet. |
| M1: scrub, extract, classify, review CLI | Not started |
| M2: second source | Not started |
| M3: permission-gated forum adapter | Not started |
| M4: dataset card and private HF push | Not started |

The fetcher for `christianity-stackexchange` is deliberately absent. CLAUDE.md ("Taxonomy First") requires `taxonomy/issues.yaml` to be human-authored before any fetcher is written. The shipped taxonomy is a machine-seeded draft carrying `approved: false`, and `Taxonomy.require_approved()` blocks pipeline stages until a human flips it. Review it, edit it, set `approved: true`, and the fetcher becomes the next unit of work.

## The M0 human checkpoint

Three decisions are waiting on a human:

1. Review and approve (or rewrite) `taxonomy/issues.yaml` and `taxonomy/registers.yaml`, then set `approved: true` in each.
2. Confirm `christianity-stackexchange` as the M0 bulk-dump source. The manifest was copied verbatim from the human-authored example in CLAUDE.md.
3. Resolve the attribution tension: rule 8 forbids usernames in `data/parsed/`, while CC BY-SA requires author attribution. The schema currently treats `author_attribution` as the single license-mandated exception to scrubbing (see `src/faithqs/schema.py`). Confirm or change that interpretation.

## Quickstart

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
cd faith-questions-dataset
uv sync
uv run pytest
uv run ruff check
```

The test suite exercises the full record model: license gating of `verbatim_text`, CC BY-SA attribution requirements, UTC normalization, PII storage gating, quarantine of invalid payloads, and referential integrity between records and the taxonomy files.

## Layout

```
CLAUDE.md               # project constitution: rules, schema, build order
taxonomy/
  issues.yaml           # DRAFT issue taxonomy (60 issues, 13 categories)
  registers.yaml        # DRAFT register vocabulary (8 registers)
sources/
  christianity-stackexchange.yaml   # M0 source manifest, human-resolved license
  permissions/          # permission correspondence (referenced by permission_ref)
src/faithqs/
  schema.py             # record model, quarantine, license and PII gates
  taxonomy.py           # taxonomy loaders, approval gate, FK checks
tests/
data/                   # gitignored in full; created at runtime
```

## Relationship to Scripture-Contextual-Retrieval

This project is independent of the contextual retrieval work and lives in this repository only because it was scaffolded from a session scoped to it. The directory is self-contained. To extract it into its own repository with history:

```bash
git subtree split --prefix=faith-questions-dataset -b faith-questions-export
# then push that branch to a new repo and add this directory to the parent's cleanup list
```

A plain copy of the directory also works, since nothing here imports from or links to the parent project.
