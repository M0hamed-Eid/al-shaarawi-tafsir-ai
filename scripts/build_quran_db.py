#!/usr/bin/env python3
"""
build_quran_db.py  —  build the canonical Quran corpus from Tanzil Uthmani.

Produces:
    data/quran/quran-uthmani.xml   (copied from pyquran for reproducibility)
    data/quran/quran.jsonl         (6236 verses: id, surah, ayah, uthmani, norm, align)
Optionally (--fts) an SQLite/FTS5 store for ad-hoc SQL queries.

Usage:
    python scripts/build_quran_db.py
    python scripts/build_quran_db.py --fts
    python scripts/build_quran_db.py --config config/enrichment.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from quran_alignment.config import load_config       # noqa: E402
from quran_alignment import quran_index as qi         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--fts", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    n = qi.build_corpus(cfg.quran_xml, cfg.quran_db_jsonl, copy_xml_to=cfg.quran_xml)
    print(f"[build] corpus: {n} verses -> {cfg.quran_db_jsonl}")
    if args.fts:
        m = qi.build_sqlite_fts(cfg.quran_xml, cfg.quran_db)
        print(f"[build] SQLite/FTS5: {m} verses -> {cfg.quran_db}")


if __name__ == "__main__":
    main()
