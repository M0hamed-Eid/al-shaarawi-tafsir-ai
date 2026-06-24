"""
evaluate.py  —  Precision / Recall / F1 against the gold set
============================================================

GOLD FORMAT (evaluation/gold.csv) — produced by build_gold.py + human review:
    episode, surah, ayah, status, confidence, start, end,
    transcript_text, canonical_quoted, VERDICT, NOTE

A row is a GROUND-TRUTH POSITIVE iff VERDICT == 1 (covers both verified
predictions AND human-added MISSED quotations). VERDICT == 0 rows are
ground-truth negatives (the system proposed them, but they are wrong).

MATCHING UNIT
-------------
We evaluate at the (episode, surah, ayah) level — the unit that matters for the
product ("did we attach the right ayah here?"). Timestamp overlap can be added
for stricter span scoring (`--require-overlap`).

DEFINITIONS
-----------
  predictions P  = system 'accepted' matches (default) for the gold episodes.
  truth      T   = gold rows with VERDICT == 1.
  TP = |P ∩ T|        (predicted AND true)
  FP = |P \ T|        (predicted but VERDICT==0 or absent from T)
  FN = |T \ P|        (true but not predicted — incl. MISSED rows)
  Precision = TP/(TP+FP)   Recall = TP/(TP+FN)   F1 = 2PR/(P+R)

The evaluator reports the confusion counts AND lists FP / FN so errors can be
inspected directly. `--status review` includes review-tier matches in P to see
the precision/recall trade-off of lowering the bar.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quran_alignment.config import load_config            # noqa: E402


def _key(episode: str, surah: str, ayah) -> tuple:
    return (str(episode).strip(), str(surah).strip(), int(ayah))


def load_gold(path: Path):
    truth, neg = set(), set()
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if not r.get("ayah"):
                continue
            k = _key(r["episode"], r["surah"], r["ayah"])
            v = (r.get("VERDICT") or "").strip()
            if v == "1":
                truth.add(k)
            elif v == "0":
                neg.add(k)
    return truth, neg


def load_predictions(enriched_dir: Path, episodes: set[str], statuses: set[str]):
    """Read predicted matches from already-enriched JSON (pipeline output)."""
    preds = set()
    for ep in episodes:
        p = enriched_dir / ep
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for m in data.get("matches", []):
            if m.get("status") in statuses:
                preds.add(_key(ep, m["surah"], m["ayah"]))
    return preds


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--gold", default="evaluation/gold.csv")
    ap.add_argument("--status", default="accepted",
                    help="comma list of statuses counted as predictions (accepted[,review])")
    ap.add_argument("--show", type=int, default=20, help="how many FP/FN to print")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    gold_path = Path(args.gold)
    if not gold_path.exists():
        sys.exit(f"gold file not found: {gold_path} (run build_gold.py then annotate)")

    truth, neg = load_gold(gold_path)
    episodes = {e for (e, _, _) in (truth | neg)}
    statuses = set(s.strip() for s in args.status.split(","))
    preds = load_predictions(cfg.output_dir, episodes, statuses)

    # restrict to the evaluated episodes
    preds = {k for k in preds if k[0] in episodes}

    tp = preds & truth
    fp = preds - truth
    fn = truth - preds

    P = len(tp) / len(preds) if preds else 0.0
    R = len(tp) / len(truth) if truth else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0

    print("=" * 60)
    print(f"Gold episodes      : {len(episodes)}")
    print(f"Truth positives    : {len(truth)}")
    print(f"Predictions ({','.join(statuses)}): {len(preds)}")
    print("-" * 60)
    print(f"TP={len(tp)}  FP={len(fp)}  FN={len(fn)}")
    print(f"Precision = {P:.3f}")
    print(f"Recall    = {R:.3f}")
    print(f"F1        = {F1:.3f}")
    print("=" * 60)
    if fp:
        print(f"\nFALSE POSITIVES (showing {min(args.show,len(fp))}):")
        for k in sorted(fp)[:args.show]:
            print(f"  {k[1]}:{k[2]}  in {k[0]}")
    if fn:
        print(f"\nFALSE NEGATIVES (showing {min(args.show,len(fn))}):")
        for k in sorted(fn)[:args.show]:
            print(f"  {k[1]}:{k[2]}  in {k[0]}")


if __name__ == "__main__":
    main()
