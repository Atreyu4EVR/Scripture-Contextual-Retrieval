"""
Orchestrator for the Book of Mormon authorship study (phase one).

Runs the 8-step workstream end to end. Steps 5-8 share a single run id so the
validation gate, analyses, and report all operate on the same run.

Steps:
  1. build_annotations.py        validate maps, compute reliability, segment index
  2. build_segments.py           5 segmentation sets
  3. detect_biblical_quotes.py   KJV n-gram quote detection + quote-removed set
  4. extract_features.py         stylometric feature tables (full + nopunct) + Delta
  5. run_validation_harness.py   KJV calibration (gate)
  6. analyze_unsupervised.py     PCA/UMAP, clustering, Delta matrices
  7. analyze_supervised.py       leave-one-chapter-out attribution
  8. generate_authorship_report.py   JSON + Markdown report + figures

Usage:
  python authorship/scripts/run_authorship_pipeline.py
  python authorship/scripts/run_authorship_pipeline.py --dry-run
  python authorship/scripts/run_authorship_pipeline.py --start-from 5
  python authorship/scripts/run_authorship_pipeline.py --quick
  python authorship/scripts/run_authorship_pipeline.py --skip-quotes

Phase one uses only in-repo texts; no API keys are required.
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

PIPELINE_STEPS = [
    {"name": "Build annotations", "script": "build_annotations.py", "run_id": False},
    {"name": "Build segments", "script": "build_segments.py", "run_id": False},
    {"name": "Detect biblical quotes", "script": "detect_biblical_quotes.py", "run_id": False,
     "skip_flag": "--skip-quotes"},
    {"name": "Extract features", "script": "extract_features.py", "run_id": False, "quick": True},
    {"name": "Validation harness (calibration)", "script": "run_validation_harness.py", "run_id": True},
    {"name": "Unsupervised analysis", "script": "analyze_unsupervised.py", "run_id": True},
    {"name": "Supervised attribution", "script": "analyze_supervised.py", "run_id": True},
    {"name": "Generate report", "script": "generate_authorship_report.py", "run_id": True},
]


def run_step(step: dict, run_id: str, args) -> bool:
    script_path = SCRIPTS_DIR / step["script"]
    if not script_path.exists():
        print(f"  Error: script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)]
    if step.get("run_id"):
        cmd += ["--run-id", run_id]
    if step.get("quick") and args.quick:
        cmd += ["--quick"]
    if step.get("skip_flag") and args.skip_quotes:
        cmd += [step["skip_flag"]]

    if args.dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return True

    print(f"  Running: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"  Completed in {elapsed:.1f}s")
        return True
    print(f"  Failed with exit code {result.returncode}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Run the Book of Mormon authorship pipeline (phase one)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--start-from", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--quick", action="store_true", help="Reduced feature vocab for a fast smoke run")
    parser.add_argument("--skip-quotes", action="store_true", help="Reuse cached KJV n-gram index")
    parser.add_argument("--run-id", default=None, help="Reuse an existing run id for steps 5-8")
    args = parser.parse_args()

    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 60)
    print("Book of Mormon Authorship Study — Phase One Pipeline")
    print("=" * 60)
    if args.dry_run:
        print("[DRY RUN MODE]")
    print(f"Run ID: {run_id}\n")
    for i, step in enumerate(PIPELINE_STEPS, 1):
        mark = " [SKIP - before start-from]" if i < args.start_from else ""
        print(f"  {i}. {step['name']}{mark}")
    print()

    failed = None
    for i, step in enumerate(PIPELINE_STEPS, 1):
        if i < args.start_from:
            continue
        print(f"\n[Step {i}/{len(PIPELINE_STEPS)}] {step['name']}")
        if not run_step(step, run_id, args) and not args.dry_run:
            failed = i
            print(f"\nPipeline stopped at step {i}. Resume with:")
            print(f"  python authorship/scripts/run_authorship_pipeline.py --start-from {i} --run-id {run_id}")
            break

    print("\n" + "=" * 60)
    if failed:
        print(f"Pipeline FAILED at step {failed}")
        sys.exit(1)
    elif args.dry_run:
        print("Dry run complete.")
    else:
        print("Pipeline completed successfully!")
        print(f"Report: authorship/results/reports/authorship_report_{run_id.replace('run_', '')}.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
