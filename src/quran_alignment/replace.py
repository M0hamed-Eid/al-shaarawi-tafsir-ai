"""
replace.py  —  Transcript replacement (timestamp-preserving)
============================================================

Turns

    "قال الله تعالى ان الله غفور رحيم ثم يكمل الشرح"

into

    "قال الله تعالى ﴿ إِنَّ اللَّهَ غَفُورٌ رَّحِيمٌ ﴾ [البقرة:173] ثم يكمل الشرح"

HOW TIMESTAMPS ARE PRESERVED
----------------------------
Every match owns a transcript word-span [word_start, word_end] in the flattened
stream, and each Word keeps its (seg_id, pos, start, end). Replacement is a pure
*string* rewrite over those spans — segment start/end timestamps are NEVER
touched, and each VerifiedMatch already carries the recited span's start/end. So
the structured output (with seconds) and the inline text stay in lock-step.

A match is rendered once, at its first word; the remaining covered words are
dropped from the enriched text only. Matches that cross a segment boundary are
attributed to the segment of their first word. Only `accepted` matches are
inlined by default (review matches stay in the structured list for a human).
"""
from __future__ import annotations

from .config import Config
from .schema import VerifiedMatch, MatchStatus
from .transcript_io import Word


def _label(m: VerifiedMatch) -> str:
    return f"[{m.surah}:{m.ayah}]"


def render_replacement(m: VerifiedMatch, cfg: Config) -> str:
    o = cfg.output
    text = m.canonical_quoted if (o.get("replace_with", "quoted") == "quoted" and m.canonical_quoted) else m.canonical_text
    return f'{o["open_marker"]}{text}{o["close_marker"]} {_label(m)}'


def build_enriched_segments(words: list[Word],
                            matches: list[VerifiedMatch],
                            transcript: dict,
                            cfg: Config) -> dict[int, str]:
    """Return {seg_id: enriched_text} for segments that contain >=1 inlined match."""
    inline = [m for m in matches if m.status == MatchStatus.accepted]
    # map global word index -> ("start", match) or ("skip", None)
    start_at: dict[int, VerifiedMatch] = {}
    covered: set[int] = set()
    for m in inline:
        start_at[m.word_start] = m
        covered.update(range(m.word_start, m.word_end + 1))

    # group match-stream words by segment, preserving order
    by_seg: dict[int, list[Word]] = {}
    for w in words:
        by_seg.setdefault(w.seg_id, []).append(w)

    touched = {m.word_start for m in inline}
    seg_ids_touched = {words[g].seg_id for g in touched} if touched else set()

    enriched: dict[int, str] = {}
    for seg_id in seg_ids_touched:
        toks: list[str] = []
        for w in by_seg.get(seg_id, []):
            if w.g in start_at:
                toks.append(render_replacement(start_at[w.g], cfg))
            elif w.g in covered:
                continue  # inside a match (covered by its first word's render)
            else:
                toks.append(w.raw)
        enriched[seg_id] = " ".join(t.strip() for t in toks if t.strip())
    return enriched


def replace_text(text_words: list[str], matches, cfg) -> str:
    """Convenience for the demo / tests: rewrite a single flat word list.
    `matches` is a list of (word_start, word_end, canonical_text, label)."""
    start_at = {ws: (we, can, lab) for ws, we, can, lab in matches}
    covered = set()
    for ws, we, _, _ in matches:
        covered.update(range(ws, we + 1))
    out = []
    i = 0
    while i < len(text_words):
        if i in start_at:
            we, can, lab = start_at[i]
            out.append(f'{cfg.output["open_marker"]}{can}{cfg.output["close_marker"]} {lab}')
            i = we + 1
        else:
            out.append(text_words[i])
            i += 1
    return " ".join(out)
