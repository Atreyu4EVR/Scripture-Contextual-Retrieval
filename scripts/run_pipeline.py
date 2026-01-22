"""
Full Pipeline Runner for LDS Standard Works Contextual Retrieval

Orchestrates the complete pipeline from raw scriptures to indexed vectors:
1. Generate chapter summaries (GPT-5.2)
2. Create contextualized verses
3. Generate embeddings (text-embedding-3-large)
4. Upsert to Pinecone

Usage:
    python scripts/run_pipeline.py              # Run full pipeline
    python scripts/run_pipeline.py --skip-summaries  # Skip if summaries exist
    python scripts/run_pipeline.py --dry-run    # Show what would run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent

PIPELINE_STEPS = [
    {
        "name": "Generate Chapter Summaries",
        "script": "generate_chapter_summaries.py",
        "description": "Generate AI-powered chapter summaries using GPT-5.2",
        "output_dir": PROJECT_ROOT / "scriptures" / "contextualized" / "summaries",
        "skip_flag": "--skip-summaries",
    },
    {
        "name": "Create Contextualized Verses",
        "script": "generate_contextualized_verses.py",
        "description": "Combine chapter context with verse text",
        "output_dir": PROJECT_ROOT / "scriptures" / "contextualized",
    },
    {
        "name": "Generate Embeddings",
        "script": "generate_embeddings.py",
        "description": "Create embeddings using text-embedding-3-large",
        "output_dir": PROJECT_ROOT / "scriptures" / "embeddings",
    },
    {
        "name": "Upsert to Pinecone",
        "script": "upsert_to_pinecone.py",
        "description": "Upload vectors to Pinecone index",
        "output_dir": None,
    },
]


def check_environment():
    """Verify required environment variables are set."""
    from dotenv import load_dotenv
    import os

    load_dotenv()

    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        print("Error: Missing required environment variables:")
        for var in missing:
            print(f"  - {var}")
        print("\nPlease set these in your .env file. See .env.example for reference.")
        return False
    return True


def run_step(step: dict, dry_run: bool = False) -> bool:
    """Run a single pipeline step."""
    script_path = SCRIPTS_DIR / step["script"]

    if not script_path.exists():
        print(f"  Error: Script not found: {script_path}")
        return False

    if dry_run:
        print(f"  [DRY RUN] Would run: python {script_path}")
        return True

    print(f"  Running: python {script_path}")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=False,
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"  Completed in {elapsed:.1f}s")
            return True
        else:
            print(f"  Failed with exit code {result.returncode}")
            return False

    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def should_skip_step(step: dict, args: argparse.Namespace) -> bool:
    """Determine if a step should be skipped based on args and existing output."""
    skip_flag = step.get("skip_flag")

    if skip_flag and getattr(args, skip_flag.lstrip("-").replace("-", "_"), False):
        output_dir = step.get("output_dir")
        if output_dir and output_dir.exists() and any(output_dir.iterdir()):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the full LDS Standard Works Contextual Retrieval pipeline"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )
    parser.add_argument(
        "--skip-summaries",
        action="store_true",
        help="Skip summary generation if summaries already exist",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        choices=[1, 2, 3, 4],
        default=1,
        help="Start from step N (1=summaries, 2=contextualize, 3=embeddings, 4=upsert)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LDS Standard Works - Contextual Retrieval Pipeline")
    print("=" * 60)

    if args.dry_run:
        print("[DRY RUN MODE - No changes will be made]\n")

    if not args.dry_run and not check_environment():
        sys.exit(1)

    print(f"\nPipeline Steps:")
    for i, step in enumerate(PIPELINE_STEPS, 1):
        status = ""
        if i < args.start_from:
            status = " [SKIP - before start-from]"
        elif should_skip_step(step, args):
            status = " [SKIP - output exists]"
        print(f"  {i}. {step['name']}{status}")
    print()

    failed_step = None
    for i, step in enumerate(PIPELINE_STEPS, 1):
        if i < args.start_from:
            continue

        print(f"\n[Step {i}/{len(PIPELINE_STEPS)}] {step['name']}")
        print(f"  {step['description']}")

        if should_skip_step(step, args):
            print("  Skipping (output already exists)")
            continue

        success = run_step(step, dry_run=args.dry_run)

        if not success and not args.dry_run:
            failed_step = i
            print(f"\nPipeline stopped at step {i}. Fix the issue and re-run with:")
            print(f"  python scripts/run_pipeline.py --start-from {i}")
            break

    print("\n" + "=" * 60)
    if failed_step:
        print(f"Pipeline FAILED at step {failed_step}")
        sys.exit(1)
    elif args.dry_run:
        print("Dry run complete. No changes were made.")
    else:
        print("Pipeline completed successfully!")
        print("\nNext steps:")
        print("  - Run evaluation: python scripts/run_evaluation.py")
        print("  - Test queries: python scripts/query_pinecone.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
