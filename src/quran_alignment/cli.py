"""
cli.py  —  command-line entry point
===================================

    python -m quran_alignment.cli build            # build the Quran corpus (JSONL)
    python -m quran_alignment.cli build --fts      # also build optional SQLite/FTS5
    python -m quran_alignment.cli demo             # run the brief's example end-to-end
    python -m quran_alignment.cli enrich-file PATH # one transcript -> enriched JSON (stdout)
    python -m quran_alignment.cli enrich-all       # batch over data/transcripts/**
    python -m quran_alignment.cli enrich-all --jobs 8 --limit 20

All commands accept `--config PATH` (defaults to config/enrichment.yaml).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from . import quran_index as qi


def _build(args):
    cfg = load_config(args.config)
    n = qi.build_corpus(cfg.quran_xml, cfg.quran_db_jsonl, copy_xml_to=cfg.quran_xml)
    print(f"[build] corpus written: {n} verses -> {cfg.quran_db_jsonl}")
    if args.fts:
        m = qi.build_sqlite_fts(cfg.quran_xml, cfg.quran_db)
        print(f"[build] optional SQLite/FTS5: {m} verses -> {cfg.quran_db}")


def _demo(args):
    from .enrich import Enricher
    enr = Enricher(load_config(args.config))
    words = "قال الله تعالى ان الله غفور رحيم ثم يكمل الشرح".split()
    demo = {"surah": "البقرة", "surah_order": 2, "episode": 1, "source_audio": "demo",
            "segments": [{"id": 0, "start": 120.0, "end": 130.0, "text": " ".join(words),
                          "words": [{"word": w, "start": 120.0 + i, "end": 121.0 + i,
                                     "probability": 0.9} for i, w in enumerate(words)]}]}
    res = enr.enrich_transcript(demo)
    for m in res.matches:
        print(json.dumps({"surah": m.surah, "ayah": m.ayah, "canonical_text": m.canonical_quoted,
                          "confidence": m.confidence, "start": m.start, "end": m.end,
                          "status": m.status.value}, ensure_ascii=False, indent=2))
    print("\nENRICHED:", res.segments[0].enriched_text)


def _enrich_file(args):
    from .enrich import Enricher
    enr = Enricher(load_config(args.config))
    res = enr.enrich_file(Path(args.path))
    out = res.model_dump_json(indent=2)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"[enrich-file] -> {args.out}  ({res.accepted} accepted, {res.review} review)")
    else:
        sys.stdout.write(out + "\n")


def _enrich_all(args):
    from . import pipeline
    cfg = load_config(args.config)
    stats = pipeline.run(cfg, cfg_path=args.config, jobs=args.jobs, limit=args.limit)
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:1500])
    print(f"\n[enrich-all] {stats['ok']}/{stats['episodes']} ok in {stats['seconds']}s "
          f"| matches: {stats['total_matches']} "
          f"(accepted {stats['accepted']}, review {stats['review']}, cross-surah {stats['cross_surah']})")


def main(argv=None):
    p = argparse.ArgumentParser(prog="quran_alignment", description="Quran reference enrichment")
    p.add_argument("--config", default=None, help="path to enrichment.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build the Quran corpus")
    b.add_argument("--fts", action="store_true", help="also build optional SQLite/FTS5 store")
    b.set_defaults(func=_build)

    d = sub.add_parser("demo", help="run the brief's example")
    d.set_defaults(func=_demo)

    e = sub.add_parser("enrich-file", help="enrich a single transcript")
    e.add_argument("path")
    e.add_argument("--out", default=None)
    e.set_defaults(func=_enrich_file)

    a = sub.add_parser("enrich-all", help="batch enrich every transcript")
    a.add_argument("--jobs", type=int, default=1)
    a.add_argument("--limit", type=int, default=None)
    a.set_defaults(func=_enrich_all)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
