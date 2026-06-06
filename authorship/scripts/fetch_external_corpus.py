"""
Phase 2 — Step A: Fetch the external comparison corpus (public-domain texts).

Downloads each external author source named in config/external_corpus.json to
authorship/data/external/raw/<key>.txt and records provenance/license in a manifest.
In-repo sources (Joseph Smith = D&C, KJV control) are not downloaded; they are read
from the repo's source JSON at the segmentation step.

Retries with exponential backoff on network errors. Skips files already present
unless --force.

Run: python authorship/scripts/fetch_external_corpus.py [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac

RAW_DIR = ac.DATA_DIR / "external" / "raw"
UA = "Mozilla/5.0 (research; scripture-authorship-study)"


def fetch_url(url: str, timeout: int = 120) -> bytes:
    last = None
    for attempt, delay in enumerate([0, 2, 4, 8, 16], start=1):
        if delay:
            time.sleep(delay)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            if len(data) > 2000:  # guard against error stubs
                return data
            last = f"too small ({len(data)} bytes)"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        print(f"    attempt {attempt} failed: {last}")
    raise RuntimeError(f"failed to fetch {url}: {last}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg = ac.load_json(ac.CONFIG_DIR / "external_corpus.json")
    fetch_tpl = cfg["fetch"]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Phase 2 Step A: Fetch external corpus")
    print("=" * 60)

    manifest = {"fetched": [], "in_repo": [], "gaps": cfg.get("gaps", [])}
    for a in cfg["authors"]:
        key, src = a["key"], a["source"]
        if src["type"] == "in_repo_volume":
            print(f"  {a['display']:28s} in-repo volume '{src['volume']}' (no download)")
            manifest["in_repo"].append({"key": key, "volume": src["volume"],
                                        "role": a["role"], "license": a["license"]})
            continue

        out = RAW_DIR / f"{key}.txt"
        if out.exists() and not args.force:
            print(f"  {a['display']:28s} cached ({out.stat().st_size:,} bytes)")
            manifest["fetched"].append({"key": key, "bytes": out.stat().st_size,
                                        "role": a["role"], "provenance": a["provenance"]})
            continue

        if src["type"] == "archive_txt":
            url = fetch_tpl["archive_txt_url"].format(id=src["id"])
        elif src["type"] == "gutenberg":
            url = fetch_tpl["gutenberg_url"].format(id=src["id"])
        else:
            print(f"  {a['display']}: unknown source type {src['type']}; skipping")
            continue

        print(f"  {a['display']:28s} fetching {url}")
        data = fetch_url(url)
        out.write_bytes(data)
        print(f"    saved {len(data):,} bytes -> {out}")
        manifest["fetched"].append({"key": key, "bytes": len(data),
                                    "role": a["role"], "provenance": a["provenance"]})

    ac.write_json(ac.DATA_DIR / "external" / "fetch_manifest.json", manifest)
    print(f"\nFetched {len(manifest['fetched'])} external source(s); "
          f"{len(manifest['in_repo'])} in-repo source(s); "
          f"{len(manifest['gaps'])} documented gap(s).")


if __name__ == "__main__":
    main()
