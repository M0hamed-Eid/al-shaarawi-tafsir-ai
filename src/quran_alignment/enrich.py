"""
enrich.py  —  orchestrator
==========================

Wires the stages into one object that turns a Whisper transcript (dict OR path)
into a fully-populated EnrichedTranscript:

    flatten (transcript_io)            STAGE A normalisation happens here
        -> collect_raw_matches (retrieval)   STAGE B + localisation (C)
        -> decide (decide)                   STAGE E (cross-surah, ambiguity)
        -> chain_and_extend (multiverse)     STAGE D (multi-verse, force-align)
        -> build_enriched_segments (replace) inline ﴿…﴾ rewrite
        -> EnrichedTranscript (schema)

The QuranIndex (corpus + bigram index + BM25) is built ONCE and reused across
all 1214 episodes — construct one Enricher, call enrich_* many times.
"""
from __future__ import annotations

from pathlib import Path

from .config import Config, load_config
from .quran_index import QuranIndex
from .retrieval import collect_raw_matches
from .decide import decide
from .multiverse import chain_and_extend
from .replace import build_enriched_segments
from .schema import EnrichedTranscript, EnrichedSegment, MatchStatus
from .transcript_io import flatten_words, load_transcript


class Enricher:
    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or load_config()
        self.index = QuranIndex(self.cfg.quran_db_jsonl)

    # -- core ----------------------------------------------------------------
    def enrich_transcript(self, transcript: dict) -> EnrichedTranscript:
        cfg = self.cfg
        norm_opts = cfg.norm
        primary_no = int(transcript.get("surah_order", 0))

        words = flatten_words(transcript, norm_opts)
        raw = collect_raw_matches(self.index, words, cfg)
        verified = decide(raw, self.index, cfg, primary_no)
        # keep accepted + (optionally) review for chaining; drop rejected
        keep = [m for m in verified if m.status != MatchStatus.rejected]
        verified = chain_and_extend(keep, words, self.index, cfg)

        if not cfg.output.get("keep_review_matches", True):
            verified = [m for m in verified if m.status == MatchStatus.accepted]

        enriched_text = {}
        if cfg.output.get("write_enriched_text", True):
            enriched_text = build_enriched_segments(words, verified, transcript, cfg)

        # build per-segment view
        seg_models: list[EnrichedSegment] = []
        matches_by_seg: dict[int, list] = {}
        for m in verified:
            seg_id = words[m.word_start].seg_id
            matches_by_seg.setdefault(seg_id, []).append(m)
        for seg in transcript.get("segments", []):
            sid = seg.get("id", 0)
            seg_models.append(EnrichedSegment(
                id=sid, start=float(seg.get("start", 0.0)), end=float(seg.get("end", 0.0)),
                text=seg.get("text", ""),
                enriched_text=enriched_text.get(sid),
                matches=sorted(matches_by_seg.get(sid, []), key=lambda x: x.word_start),
            ))

        out = EnrichedTranscript(
            surah=transcript.get("surah", ""),
            surah_order=primary_no,
            episode=int(transcript.get("episode", 0)),
            source_transcript=transcript.get("source_audio", ""),
            primary_surah_no=primary_no,
            matches=verified,
            segments=seg_models,
        ).recount()
        return out

    def enrich_file(self, path: Path) -> EnrichedTranscript:
        return self.enrich_transcript(load_transcript(Path(path)))
