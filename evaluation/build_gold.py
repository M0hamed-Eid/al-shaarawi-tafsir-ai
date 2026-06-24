"""
build_gold.py  —  bootstrap a gold-standard validation set
==========================================================

Annotating Quran references from scratch is slow. It is MUCH faster to let the
pipeline propose matches and have a human *verify* them, then separately scan
for misses. This script samples episodes, runs enrichment, and writes a CSV
annotation sheet:

    columns: episode, surah, ayah, status, confidence, start, end,
             transcript_text, canonical_quoted,
             VERDICT   <- annotator fills: 1 (correct) / 0 (wrong) / blank
             NOTE

WORKFLOW to build the gold set
------------------------------
1. python evaluation/build_gold.py --sample 25 --seed 7
   -> evaluation/gold_sheet.csv  (predictions to verify)
2. A qualified annotator (knows the Quran / uses a mushaf) marks VERDICT for
   every row (1 correct, 0 wrong).
3. CRITICAL for RECALL: while listening/reading each sampled episode, the
   annotator ALSO appends rows for any Quran quotation the system MISSED, with
   VERDICT=1 and status=MISSED (leave timestamps if known, else blank).
4. Save as evaluation/gold.csv. Then run evaluate.py.

Sampling is stratified by surah so short and long surahs are both represented.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quran_alignment.config import load_config            # noqa: E402
from quran_alignment.enrich import Enricher               # noqa: E402
from quran_alignment.pipeline import discover             # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--sample", type=int, default=25, help="episodes to sample")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="evaluation/gold_sheet.csv")
    ap.add_argument("--include-review", action="store_true",
                    help="also list 'review' matches (recommended for tuning)")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    files = discover(cfg.transcripts_dir)
    random.seed(args.seed)
    # stratify by surah (parent dir)
    by_surah: dict[str, list[Path]] = {}
    for f in files:
        by_surah.setdefault(f.parent.name, []).append(f)
    picks: list[Path] = []
    surahs = list(by_surah)
    random.shuffle(surahs)
    i = 0
    while len(picks) < min(args.sample, len(files)):
        s = surahs[i % len(surahs)]
        bucket = by_surah[s]
        if bucket:
            picks.append(bucket.pop(random.randrange(len(bucket))))
        i += 1

    enr = Enricher(cfg)
    rows = []
    for f in picks:
        res = enr.enrich_file(f)
        for m in res.matches:
            if m.status.value == "rejected":
                continue
            if m.status.value == "review" and not args.include_review:
                continue
            rows.append([f.relative_to(cfg.transcripts_dir), m.surah, m.ayah,
                         m.status.value, m.confidence, m.start, m.end,
                         m.transcript_text, m.canonical_quoted, "", ""])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["episode", "surah", "ayah", "status", "confidence", "start",
                    "end", "transcript_text", "canonical_quoted", "VERDICT", "NOTE"])
        w.writerows(rows)
    print(f"[build_gold] {len(picks)} episodes, {len(rows)} predictions -> {out}")
    print("Annotate VERDICT (1/0) and ADD missed quotations (status=MISSED, VERDICT=1).")


if __name__ == "__main__":
    main()
