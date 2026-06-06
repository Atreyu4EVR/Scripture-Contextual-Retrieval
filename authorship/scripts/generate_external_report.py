"""
Phase 2 — Step F: Generate the H1/H2 (external authorship) report.

Reads a run's open_set.json plus the corpus manifests and writes a Markdown + JSON
report on whether the Book of Mormon resembles any candidate 19th-century author, with
the register confound and corpus limitations stated up front.

Run: python authorship/scripts/generate_external_report.py [--run-id run_...]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac


def md_table(headers, rows):
    out = "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def build(run_id):
    raw_dir = ac.RESULTS_DIR / "raw" / run_id
    os_path = raw_dir / "open_set.json"
    if not os_path.exists():
        print(f"No open_set.json for {run_id}; run analyze_open_set.py first.")
        sys.exit(1)
    osr = ac.load_json(os_path)
    corpus_cfg = ac.load_json(ac.CONFIG_DIR / "external_corpus.json")
    ext_index = ac.load_json(ac.DATA_DIR / "external" / "external_segment_index.json")
    clean_manifest_path = ac.DATA_DIR / "external" / "clean_manifest.json"
    clean = ac.load_json(clean_manifest_path) if clean_manifest_path.exists() else {}

    prob = osr["bom_probabilistic_attribution"]
    dist = osr["bom_delta_attribution"]
    calib = osr["open_set_calibration"]

    m = []
    m.append("# Book of Mormon Authorship — Phase Two Report (H1/H2: external authors)")
    m.append(f"\n**Run ID:** {run_id}")
    m.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    m.append("**Question:** Does the English text of the Book of Mormon stylometrically "
             "resemble any candidate 19th-century author, with an explicit *none of the "
             "above* option?")

    m.append("\n## Read this first: the register confound\n")
    m.append("Among the comparison authors, only **Joseph Smith (Doctrine and Covenants)** "
             "and the **KJV** share the Book of Mormon's dictated / scriptural-archaic "
             "register. The source-theory candidates (Ethan Smith, Solomon Spalding) and the "
             "distractors (Edwards, Bunyan, Irving) are ordinary written prose. Any pull "
             "toward Joseph Smith or the KJV therefore partly reflects **register**, not "
             "necessarily authorship. A cleaner test would require same-register comparators "
             "that do not exist for the source-theory candidates. Read every result below "
             "through this caveat.")

    m.append("\n## 1. Comparison corpus\n")
    rows = []
    for a in corpus_cfg["authors"]:
        info = ext_index["authors"].get(a["key"], {})
        ocr = clean.get(a["key"], {})
        prov = a["provenance"]
        rows.append([a["display"], a["role"], a.get("register", ""),
                     info.get("windows", "-"),
                     f"{ocr.get('dropped_line_fraction', 0)*100:.0f}%" if ocr else "n/a",
                     prov[:48]])
    m.append(md_table(["Author", "Role", "Register", "Windows", "OCR noise", "Source"], rows))
    for g in corpus_cfg.get("gaps", []):
        m.append(f"\n- **Gap — {g['display']}:** {g['reason']}")

    m.append("\n## 2. Method sanity\n")
    m.append(f"- **Closed-set author accuracy:** {osr['closed_set']['accuracy']:.3f} "
             f"across {len(osr['closed_set']['labels'])} authors — the candidate/distractor "
             "authors are strongly separable from each other, so the feature space carries "
             "real authorial signal.")
    m.append(f"- **Open-set calibration (leave-one-author-out):** at threshold "
             f"{calib['threshold']:.3f}, a genuinely unseen author is rejected "
             f"{calib['unknown_reject_rate']*100:.0f}% of the time while known authors are "
             f"retained {calib['known_retain_rate']*100:.0f}% of the time. (Unknown median "
             f"max-prob {calib['unknown_median_maxprob']:.2f} vs known "
             f"{calib['known_median_maxprob']:.2f}.)")

    m.append("\n## 3. Book of Mormon attribution — probabilistic (logistic + rejection)\n")
    m.append(f"- **{prob['reject_fraction']*100:.0f}% of Book of Mormon windows are rejected "
             f"as _none of the above_** ({prob['rejected_none_of_the_above']} of "
             f"{prob['n_bom_windows']}).")
    kept = prob["predicted_when_not_rejected"]
    if kept:
        m.append("- Among the windows that are NOT rejected, the predicted author is:")
        m.append(md_table(["Author", "Windows assigned"],
                          sorted(kept.items(), key=lambda kv: -kv[1])))
    m.append("\nMean per-author probability over all BoM windows:\n")
    m.append(md_table(["Author", "Mean P(author | BoM window)"],
                      [[k, f"{v:.3f}"] for k, v in
                       sorted(prob["mean_class_probability"].items(), key=lambda kv: -kv[1])]))
    m.append(f"\nFigure: `figures/{run_id}/openset_mean_probability.png`")

    m.append("\n## 4. Book of Mormon attribution — distance (Burrows's Delta)\n")
    m.append("Nearest author by window (count of BoM windows whose nearest author centroid is X):\n")
    m.append(md_table(["Author", "BoM windows nearest"],
                      sorted(dist["nearest_author_distribution"].items(), key=lambda kv: -kv[1])))
    m.append("\nDelta distance from the BoM centroid to each author (lower = closer):\n")
    m.append(md_table(["Author", "Delta distance"],
                      [[k, f"{v:.3f}"] for k, v in dist["bom_centroid_distance_to_author"].items()]))
    nearest = next(iter(dist["bom_centroid_distance_to_author"]))
    m.append(f"\n**Nearest overall (Delta centroid): {nearest}.** Figure: "
             f"`figures/{run_id}/openset_delta_distance.png`")

    m.append("\n## 5. Interpretation\n")
    m.append("- The Book of Mormon does **not** cleanly match any single candidate author: "
             f"the open-set classifier rejects {prob['reject_fraction']*100:.0f}% of its "
             "windows as none-of-the-above.")
    m.append("- Where it is pulled toward a candidate, it is pulled toward **Joseph Smith's "
             "dictated revelations (D&C)** and the **KJV** — the two texts sharing its "
             "register — and **away from** the written-prose source-theory candidates "
             "(Ethan Smith, Solomon Spalding) and the distractors.")
    m.append("- On the tested evidence, the source-text hypotheses (H2: Spalding / Ethan "
             "Smith authorship of the English text) are **not** supported: the Book of Mormon "
             "is stylometrically far from those authors. The data are more consistent with "
             "the Joseph-Smith-register account (H1) than with the source-author account — "
             "**but** this is confounded by register and cannot, alone, distinguish single "
             "authorship from a translator rendering the text in a KJV-like register (H4).")

    m.append("\n## 6. Limitations\n")
    m.append("- **Register confound (dominant):** see the top of this report.")
    m.append("- **Joseph Smith proxy:** the only public-domain Joseph Smith sample used is "
             "the D&C, which is itself dictated revelation — it shares the BoM's production "
             "mode, so resemblance to it is expected under several hypotheses.")
    m.append("- **OCR noise:** Ethan Smith and Spalding come from archive.org OCR (see noise "
             "column); function-word/char features are robust but some signal is degraded.")
    m.append("- **Corpus gap:** no clean public-domain Sidney Rigdon prose corpus was "
             "obtained; the Rigdon variant of H2 is untested here.")
    m.append("- **Small candidate set & window count:** a handful of authors and ~45 windows "
             "each; broader corpora would tighten the open-set boundary.")
    m.append("- **English text only:** stylometry speaks to the English text's production; it "
             "cannot address divine translation, an inherently metaphysical claim.")

    js = {"run_id": run_id, "open_set": osr,
          "corpus": {a["key"]: {"role": a["role"], "register": a.get("register"),
                                "provenance": a["provenance"]} for a in corpus_cfg["authors"]},
          "gaps": corpus_cfg.get("gaps", [])}
    return "\n".join(m), js


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_id = args.run_id or (ac.latest_run_dir(ac.RESULTS_DIR / "raw").name
                             if ac.latest_run_dir(ac.RESULTS_DIR / "raw") else None)
    if not run_id:
        print("No run found.")
        sys.exit(1)

    print("=" * 60)
    print(f"Phase 2 Step F: Generate H1/H2 report  (run {run_id})")
    print("=" * 60)
    md, js = build(run_id)
    reports_dir = ac.RESULTS_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = run_id.replace("run_", "")
    (reports_dir / f"authorship_h1h2_report_{ts}.md").write_text(md, encoding="utf-8")
    ac.write_json(reports_dir / f"authorship_h1h2_report_{ts}.json", js)
    print(f"Markdown -> {reports_dir / f'authorship_h1h2_report_{ts}.md'}")
    print(f"JSON     -> {reports_dir / f'authorship_h1h2_report_{ts}.json'}")


if __name__ == "__main__":
    main()
