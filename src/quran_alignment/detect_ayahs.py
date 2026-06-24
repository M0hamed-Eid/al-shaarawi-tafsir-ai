"""
detect_ayahs.py  —  backward-compatible convenience shim.

Thin helper kept for older call-sites: returns the list of VerifiedMatch for a
single transcript file.

    from quran_alignment.detect_ayahs import detect
    matches = detect("data/transcripts/البقرة/ep_001.json")
"""
from pathlib import Path

from .enrich import Enricher
from .config import load_config


def detect(transcript_path, config=None):
    enr = Enricher(load_config(config))
    return enr.enrich_file(Path(transcript_path)).matches


__all__ = ["detect"]
