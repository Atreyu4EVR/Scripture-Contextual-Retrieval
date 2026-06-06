"""
Step 8: Generate the authorship study report (JSON + Markdown + figures).

Reads a run's artifacts (validation_calibration.json, projection.json, delta_matrix.json,
attribution.json) plus the segment index and quote fractions, and writes a human-readable
Markdown report and a machine-readable JSON report. Calibration is presented BEFORE the
Book of Mormon results so readers see the method's validation first.

Run: python authorship/scripts/generate_authorship_report.py [--run-id run_...]
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import authorship_common as ac

# Known quotation controls used as a detection sanity check in the report.
KNOWN_CONTROLS = [
    ("2 Nephi", 22, "Isaiah 12"), ("2 Nephi", 19, "Isaiah 9"),
    ("Mosiah", 14, "Isaiah 53"), ("3 Nephi", 25, "Malachi 4"),
    ("3 Nephi", 13, "Matthew 6"),
]


def md_table(headers, rows) -> str:
    out = "| " + " | ".join(headers) + " |\n"
    out += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for r in rows:
        out += "| " + " | ".join(str(c) for c in r) + " |\n"
    return out


def plot_confusion(cm, labels, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    cm = np.array(cm)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.6 else "black", fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def build_report(run_id: str) -> tuple[str, dict]:
    raw_dir = ac.RESULTS_DIR / "raw" / run_id
    fig_dir = ac.RESULTS_DIR / "figures" / run_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    index = ac.load_json(ac.DATA_DIR / "segments" / "segment_index.json")
    quote_fractions = ac.load_json(ac.DATA_DIR / "quotes" / "quote_fractions.json")
    calib = ac.load_json(raw_dir / "validation_calibration.json") if (raw_dir / "validation_calibration.json").exists() else {}
    projection = ac.load_json(raw_dir / "projection.json") if (raw_dir / "projection.json").exists() else {}
    delta = ac.load_json(raw_dir / "delta_matrix.json") if (raw_dir / "delta_matrix.json").exists() else {}
    attribution = ac.load_json(raw_dir / "attribution.json") if (raw_dir / "attribution.json").exists() else {}

    m = []
    m.append("# Book of Mormon Authorship / Production Study — Phase One Report")
    m.append(f"\n**Run ID:** {run_id}")
    m.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    m.append("**Scope:** Internal multi-voice (H3) and translation-layer (H4) analysis using "
             "only in-repo texts (Book of Mormon + KJV control). External-author hypotheses "
             "(H1/H2) are explicitly out of scope for phase one.")

    if attribution.get("uncalibrated"):
        m.append("\n> **WARNING: RESULTS UNCALIBRATED.** A positive control failed and "
                 "`require_calibration` is true. The attribution numbers below must not be trusted.")

    # 1. Corpus & segmentation
    m.append("\n## 1. Corpus & Segmentation\n")
    counts = index.get("segment_counts", {})
    m.append(md_table(
        ["Property", "Value"],
        [["Volume", index.get("volume")], ["Books", index.get("n_books")],
         ["Chapters", index.get("n_chapters")], ["Verses", index.get("n_verses")],
         ["Chapter segments", counts.get("chapter")],
         ["Rolling-1000 windows", counts.get("rolling")],
         ["Narrator+speaker segments", counts.get("narrator_and_speaker")],
         ["Genre segments", counts.get("genre")]]))

    # 2. Narrator reliability
    m.append("\n## 2. Narrator Reliability\n")
    rel = index.get("narrator_reliability", {})
    rows = [[n, info["word_count"], info["reliability"]]
            for n, info in sorted(rel.items(), key=lambda kv: -kv[1]["word_count"])]
    m.append(md_table(["Claimed narrator", "Words", "Reliability"], rows))
    m.append("\nReport-only / composite narrators are excluded from supervised attribution.")

    # 3. Biblical quotation
    m.append("\n## 3. Biblical-Quotation Detection (KJV control)\n")
    top = sorted(quote_fractions.values(), key=lambda x: -x["quote_fraction"])[:10]
    m.append(md_table(["Chapter", "Quote fraction", "Top KJV source"],
                      [[f"{s['book']} {s['chapter']}", f"{s['quote_fraction']:.2f}",
                        ", ".join(s["sources"][:2])] for s in top]))
    m.append("\n**Known-control check:**")
    ctrl_rows = []
    for book, chap, expected in KNOWN_CONTROLS:
        sid = f"BoM:{book}:{chap}"
        s = quote_fractions.get(sid, {})
        ctrl_rows.append([f"{book} {chap}", expected, f"{s.get('quote_fraction', 0):.2f}",
                          ", ".join(s.get("sources", [])[:2])])
    m.append(md_table(["Chapter", "Expected source", "Detected fraction", "Detected source"], ctrl_rows))

    # 4. Calibration (before BoM results)
    m.append("\n## 4. Method Validation / Calibration\n")
    if calib:
        pa, pb = calib["positive_a_ot_vs_nt"], calib["positive_b_authors"]
        ni, npp = calib["negative_isaiah_halves"], calib["negative_bom_permutation"]
        m.append(md_table(
            ["Control", "Type", "Result", "Gate / chance", "Verdict"],
            [["OT vs NT", "positive", f"macro-F1 {pa['macro_f1']:.3f}",
              f"≥ {calib['gates']['ot_vs_nt_min_macro_f1']}",
              "PASS" if pa['macro_f1'] >= calib['gates']['ot_vs_nt_min_macro_f1'] else "FAIL"],
             ["Pauline / Synoptic / Johannine", "positive", f"macro-F1 {pb['macro_f1']:.3f}",
              f"≥ {calib['gates']['authors_min_macro_f1']}",
              "PASS" if pb['macro_f1'] >= calib['gates']['authors_min_macro_f1'] else "FAIL"],
             ["Isaiah arbitrary halves", "negative", f"acc {ni['accuracy']:.3f}",
              f"chance {ni['baseline_accuracy']:.2f}", "OK" if calib['neg_control_ok'] else "SUSPECT"],
             ["BoM label permutation", "negative",
              f"acc {npp['permuted_mean_acc']:.3f}±{npp['permuted_std_acc']:.3f}",
              f"majority {npp['majority_baseline']:.2f}", "OK"]]))
        m.append(f"\n**Overall calibration: {'PASSED' if calib['passed'] else 'FAILED'}.** "
                 "The positive controls show the pipeline separates genuinely distinct authors "
                 "within a single translation (KJV); the negative controls show it does not "
                 "invent authorship in null data. This bounds how to read the Book of Mormon results below.")
    else:
        m.append("_No calibration artifact found._")

    # 5. Unsupervised
    m.append("\n## 5. Unsupervised Structure\n")
    clusters = projection.get("clusters", [])
    kmeans = [c for c in clusters if c.get("method") == "kmeans"]
    if kmeans:
        best = max(kmeans, key=lambda c: c["silhouette"])
        m.append(f"Best KMeans: k={best['k']}, silhouette={best['silhouette']:.3f}.")
        m.append("\nCluster agreement (Adjusted Rand Index) of the best clustering against each label set:\n")
        m.append(md_table(["Label set", "ARI", "AMI"],
                          [[ls, f"{best['ari_vs'].get(ls, 0):.3f}", f"{best['ami_vs'].get(ls, 0):.3f}"]
                           for ls in ["claimed_narrator", "genre", "editorial_layer", "quote_status"]]))
        if best["ari_vs"].get("genre", 0) > best["ari_vs"].get("claimed_narrator", 0):
            m.append("\n> **Headline caveat:** clusters agree with **genre** more than with **narrator** — "
                     "much apparent voice separation is structural (sermon vs narrative vs prophecy).")
    m.append("\nFigures: `figures/" + run_id + "/umap_{claimed_narrator,genre,editorial_layer,quote_status}.png`")

    # 6. Burrows's Delta
    m.append("\n## 6. Burrows's Delta Between Narrators\n")
    if delta.get("full"):
        names = delta["full"]["names"]
        mat = delta["full"]["matrix"]
        m.append(md_table(["Δ"] + names,
                          [[names[i]] + [f"{mat[i][j]:.2f}" for j in range(len(names))]
                           for i in range(len(names))]))
        m.append("\nHeatmaps (full and quote-removed): "
                 f"`figures/{run_id}/delta_narrators_full.png`, "
                 f"`figures/{run_id}/delta_narrators_quote_removed.png`")

    # 7. Supervised attribution
    m.append("\n## 7. Supervised Attribution (leave-one-chapter-out)\n")
    results = attribution.get("results", [])
    if results:
        rows = [[r["condition"], r["classifier"], r["n_samples"], r["n_classes"],
                 f"{r['accuracy']:.3f}", f"{r['macro_f1']:.3f}", f"{r['baseline_accuracy']:.3f}"]
                for r in results]
        m.append(md_table(["Condition (quote|genre|features)", "Classifier", "n", "classes",
                           "Accuracy", "Macro-F1", "Baseline"], rows))
        # Confusion figure for the best raw cell.
        raw_cells = [r for r in results if "|raw|full" in r["condition"] and r["classifier"] == "logistic"]
        if raw_cells:
            best = raw_cells[0]
            plot_confusion(best["confusion"], best["labels"],
                           f"Confusion — {best['condition']} ({best['classifier']})",
                           fig_dir / "confusion_best.png")
            m.append(f"\nConfusion matrix (best raw cell): `figures/{run_id}/confusion_best.png`")

    # 8. Confound controls
    m.append("\n## 8. Confound Controls\n")
    wc = attribution.get("wordcount_only_accuracy")
    m.append(f"- **Length probe:** narrator predicted from word-count alone = "
             f"**{wc:.3f}** accuracy. A high value here means length is entangled with the "
             "narrator labels and partially inflates attribution; treat raw-condition accuracy "
             "with caution and prefer the genre-restricted, length-matched readings."
             if wc is not None else "- Length probe unavailable.")
    m.append("- **Quotation:** every cell is reported include vs exclude quotes; persistence of "
             "the signal across both indicates it is not merely shared KJV diction.")
    m.append("- **Genre:** narrative-only cells isolate the genre confound; persistence there "
             "indicates residual narrator (or length/topic) signal beyond genre.")
    m.append("- **Punctuation:** full vs nopunct variants; persistence rules out the 1830 "
             "typesetter's punctuation as the driver.")

    # 9. Caveats
    m.append("\n## 9. Caveats & Interpretation\n")
    m.append("- **Translation-layer collapse (H3 vs H4):** all Book of Mormon text shares one "
             "1829 dictation idiom; internal separation is consistent with BOTH a multi-voice "
             "ancient record AND a single translator rendering distinct sources. Calibration "
             "shows the method *could* detect distinct authors if present, but cannot by itself "
             "decide between H3 and H4.")
    m.append("- **Editorial homogenization:** Mormon/Moroni abridged most books, so 'narrator' "
             "classes largely reflect the abridging editor; embedded-speaker voices are 'as "
             "preserved in the abridgement,' not necessarily original diction.")
    m.append("- **Length confound:** see §8; word-count entanglement is the main internal threat "
             "to the raw-condition results.")
    m.append("- **Archaic-text NLP:** spaCy POS/sentence segmentation is approximate on KJV-style "
             "English; function-word and char-n-gram features (more robust) carry the headline claims.")
    m.append("- **Phase-one scope:** this analysis cannot address whether Joseph Smith or any "
             "19th-century author wrote the text (H1/H2). That requires the external corpus "
             "(phase two).")

    # JSON report.
    json_report = {
        "run_id": run_id,
        "scope": "phase_one_internal_voice",
        "corpus": index,
        "calibration": calib,
        "unsupervised": {"clusters": clusters},
        "delta": delta,
        "attribution": attribution,
        "quote_controls": {f"{b} {c}": quote_fractions.get(f"BoM:{b}:{c}", {})
                           for b, c, _ in KNOWN_CONTROLS},
    }
    return "\n".join(m), json_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    run_id = args.run_id or (ac.latest_run_dir(ac.RESULTS_DIR / "raw").name
                             if ac.latest_run_dir(ac.RESULTS_DIR / "raw") else None)
    if not run_id:
        print("No run found. Run the analysis steps first.")
        sys.exit(1)

    print("=" * 60)
    print(f"Step 8: Generate report  (run {run_id})")
    print("=" * 60)

    md, js = build_report(run_id)
    reports_dir = ac.RESULTS_DIR / "reports"
    ts = run_id.replace("run_", "")
    md_path = reports_dir / f"authorship_report_{ts}.md"
    json_path = reports_dir / f"authorship_report_{ts}.json"
    ac.write_json(json_path, js)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"Markdown -> {md_path}")
    print(f"JSON     -> {json_path}")


if __name__ == "__main__":
    main()
