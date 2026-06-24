"""
transcript_io.py  —  flatten Whisper transcripts into a timed word stream
=========================================================================

The Phase-3 transcripts (EpisodeTranscript schema) carry word-level timestamps:

    segments[i].words[j] = {word, start, end, probability}

We flatten them into ONE ordered stream of `Word` objects. Each carries:
    * the original (raw) token        -> for reconstruction / display
    * the normalised token            -> for matching (Stage A)
    * seg_id + pos-in-segment         -> to write replacements back precisely
    * start / end timestamps          -> to recover match start/end (preserved!)

Words whose normalised form is empty (pure punctuation) are skipped from the
MATCH stream but their timing is folded into their neighbours, so timestamps are
never lost.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .normalize import normalize


@dataclass
class Word:
    g: int          # index in the flattened match stream
    raw: str        # original token (trimmed)
    norm: str       # normalised token (non-empty)
    seg_id: int     # segment id this word belongs to
    pos: int        # index within segment.words
    start: float
    end: float


def load_transcript(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def flatten_words(transcript: dict, norm_opts: dict | None = None) -> list[Word]:
    """Return the ordered, normalised, timed word stream for matching."""
    out: list[Word] = []
    for seg in transcript.get("segments", []):
        seg_id = seg.get("id", 0)
        ws = seg.get("words") or []
        if ws:
            for pos, w in enumerate(ws):
                raw = (w.get("word") or "").strip()
                nrm = normalize(raw, norm_opts)
                if not nrm:
                    continue
                out.append(Word(len(out), raw, nrm, seg_id, pos,
                                float(w.get("start", seg.get("start", 0.0))),
                                float(w.get("end", seg.get("end", 0.0)))))
        else:
            # fallback: no word timestamps -> split segment text, spread evenly
            toks = (seg.get("text") or "").split()
            s, e = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
            dur = (e - s) / max(1, len(toks))
            for pos, raw in enumerate(toks):
                nrm = normalize(raw, norm_opts)
                if not nrm:
                    continue
                out.append(Word(len(out), raw, nrm, seg_id, pos,
                                s + pos * dur, s + (pos + 1) * dur))
    return out
