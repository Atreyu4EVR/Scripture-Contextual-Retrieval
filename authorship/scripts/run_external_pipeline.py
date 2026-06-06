"""
Phase-two orchestrator: external authorship attribution (H1/H2).

Runs the external-corpus workstream end to end. Requires phase one to have been run
first (it reuses the Book of Mormon rolling-window segments). Steps E/F share a run id.

Steps:
  A. fetch_external_corpus.py        download public-domain candidate/distractor texts
  B. clean_external_corpus.py        strip boilerplate, de-hyphenate, drop OCR garbage
  C. build_external_segments.py      balanced fixed-window segments per author
  D. extract_attribution_features.py shared feature space (authors + BoM windows)
  E. analyze_open_set.py             open-set attribution of the Book of Mormon
  F. generate_external_report.py     H1/H2 Markdown + JSON report

Usage:
  python authorship/scripts/run_external_pipeline.py
  python authorship/scripts/run_external_pipeline.py --dry-run
  python authorship/scripts/run_external_pipeline.py --start-from D
  python authorship/scripts/run_external_pipeline.py --quick

Network is required for steps A (Gutenberg + archive.org). Set --skip-fetch / cached
files to avoid re-downloading.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parents[1]

STEPS = [
    {"id": "A", "name": "Fetch external corpus", "script": "fetch_external_corpus.py", "run_id": False},
    {"id": "B", "name": "Clean external corpus", "script": "clean_external_corpus.py", "run_id": False},
    {"id": "C", "name": "Build external segments", "script": "build_external_segments.py", "run_id": False},
    {"id": "D", "name": "Extract attribution features", "script": "extract_attribution_features.py",
     "run_id": False, "quick": True},
    {"id": "E", "name": "Open-set attribution", "script": "analyze_open_set.py", "run_id": True},
    {"id": "F", "name": "Generate H1/H2 report", "script": "generate_external_report.py", "run_id": True},
]
ORDER = [s["id"] for s in STEPS]


def run_step(step, run_id, args):
    cmd = [sys.executable, str(SCRIPTS_DIR / step["script"])]
    if step.get("run_id"):
        cmd += ["--run-id", run_id]
    if step.get("quick") and args.quick:
        cmd += ["--quick"]
    if args.dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True
    print(f"  Running: {' '.join(cmd)}")
    start = time.time()
    rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
    print(f"  {'Completed' if rc == 0 else 'FAILED'} in {time.time()-start:.1f}s")
    return rc == 0


def main():
    parser = argparse.ArgumentParser(description="Run the external authorship pipeline (phase two)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-from", choices=ORDER, default="A")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    start_idx = ORDER.index(args.start_from)

    print("=" * 60)
    print("Book of Mormon Authorship — Phase Two (H1/H2) Pipeline")
    print("=" * 60)
    if args.dry_run:
        print("[DRY RUN MODE]")
    print(f"Run ID: {run_id}\n")
    for i, s in enumerate(STEPS):
        mark = "  [skip - before start-from]" if i < start_idx else ""
        print(f"  {s['id']}. {s['name']}{mark}")
    print()

    failed = None
    for i, step in enumerate(STEPS):
        if i < start_idx:
            continue
        print(f"\n[Step {step['id']}] {step['name']}")
        if not run_step(step, run_id, args) and not args.dry_run:
            failed = step["id"]
            print(f"\nStopped at step {step['id']}. Resume with:")
            print(f"  python authorship/scripts/run_external_pipeline.py --start-from {step['id']} --run-id {run_id}")
            break

    print("\n" + "=" * 60)
    if failed:
        print(f"Pipeline FAILED at step {failed}")
        sys.exit(1)
    elif args.dry_run:
        print("Dry run complete.")
    else:
        print("Phase-two pipeline completed successfully!")
        print(f"Report: authorship/results/reports/authorship_h1h2_report_{run_id.replace('run_', '')}.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
