"""
multiverse.py  —  STAGE D : multi-verse / lost-boundary detection
=================================================================

Al-Shaarawi (and reciters generally) often read 2..10 consecutive verses in one
breath; Whisper then merges them into one or more segments with NO verse
boundary markers. After Stage E we already have per-verse matches (each verse is
independently seeded by its own bigram). This stage:

  1. CHAINS consecutive verses: matches whose verses are (v, v+1, v+2, ...) in
     the SAME surah and whose transcript spans are adjacent (gap <= max_gap_words)
     are linked with a shared chain_id and the run's ayah range is recorded.

  2. EXTENDS chains by FORCED ALIGNMENT: a middle/edge verse may have been missed
     by Stage B (its only shared bigram was too common, or ASR garbled it). For
     each chain edge we force-align the neighbouring canonical verse against the
     adjoining transcript region; if it clears `extend_threshold` it is inserted,
     recovering verses the seed step skipped. This is what makes long recitations
     robust to lost boundaries.

Output: the same VerifiedMatch list, now with chain_id set and ayah ranges, plus
any verses recovered by extension. Per-verse granularity (and per-verse
timestamps) is preserved.
"""
from __future__ import annotations

from .config import Config
from .quran_index import QuranIndex
from .schema import VerifiedMatch, MatchStatus
from .transcript_io import Word
from .verify import seed_extend, compute_confidence, coverage, AlignResult


def _force_align(verse_tokens, norm, lo, hi):
    """Try to align a verse anywhere in transcript window [lo,hi) WITHOUT a
    precomputed bigram seed: probe every position that starts a shared bigram."""
    best: AlignResult | None = None
    vset = set()
    for i in range(len(verse_tokens) - 1):
        vset.add((verse_tokens[i], verse_tokens[i + 1]))
    for p in range(max(0, lo), min(len(norm) - 1, hi)):
        if (norm[p], norm[p + 1]) not in vset:
            continue
        res = seed_extend(verse_tokens, norm, p)
        if res and (best is None or res.matched_tokens > best.matched_tokens):
            best = res
    return best


def chain_and_extend(matches: list[VerifiedMatch], words: list[Word],
                     index: QuranIndex, cfg: Config) -> list[VerifiedMatch]:
    mv = cfg.multiverse
    max_gap = int(mv["max_gap_words"])
    extend_t = float(mv["extend_threshold"]) * 100.0
    max_chain = int(mv["max_chain"])
    norm = [w.norm for w in words]
    primary = matches[0].surah_no if matches else 0  # not used; kept for clarity

    # index existing matches by verse_id for quick neighbour lookup
    matches = sorted(matches, key=lambda m: m.word_start)

    # ---- 1. EXTENSION: grow runs by force-aligning neighbours ---------------
    by_verse = {m.verse_id: m for m in matches}
    added: list[VerifiedMatch] = []
    frontier = list(matches)
    guard = 0
    while frontier and guard < 5000:
        guard += 1
        m = frontier.pop()
        # try the NEXT verse in the same surah, just after this span
        nxt = index.next_verse(m.verse_id)
        if nxt and nxt.verse_id not in by_verse:
            lo = m.word_end + 1
            hi = lo + len(nxt.tokens) + max_gap
            res = _force_align(nxt.tokens, norm, lo - 1, hi)
            if res and res.fuzz_score >= extend_t and res.word_start - m.word_end - 1 <= max_gap:
                vm = _mk(nxt, res, words, m.is_cross_surah)
                by_verse[nxt.verse_id] = vm
                added.append(vm)
                frontier.append(vm)
        # try the PREVIOUS verse in the same surah, just before this span
        prev = index.get(m.verse_id - 1)
        if prev and prev.surah_no == m.surah_no and prev.verse_id not in by_verse:
            hi = m.word_start
            lo = max(0, hi - len(prev.tokens) - max_gap)
            res = _force_align(prev.tokens, norm, lo, hi + 1)
            if res and res.fuzz_score >= extend_t and m.word_start - res.word_end - 1 <= max_gap:
                vm = _mk(prev, res, words, m.is_cross_surah)
                by_verse[prev.verse_id] = vm
                added.append(vm)
                frontier.append(vm)

    allm = sorted(by_verse.values(), key=lambda m: m.word_start)

    # ---- 2. CHAINING: assign chain_id + ayah ranges -------------------------
    chain_id = 0
    i = 0
    while i < len(allm):
        j = i
        run = [allm[i]]
        while j + 1 < len(allm):
            a, b = allm[j], allm[j + 1]
            consecutive = (b.surah_no == a.surah_no and b.ayah == a.ayah + 1)
            adjacent = (0 <= b.word_start - a.word_end - 1 <= max_gap)
            if consecutive and adjacent and len(run) < max_chain:
                run.append(b)
                j += 1
            else:
                break
        if len(run) > 1:
            chain_id += 1
            a0, aN = run[0].ayah, run[-1].ayah
            for m in run:
                m.chain_id = chain_id
                m.ayah_end = aN  # every member exposes the full run's range
                # leave m.ayah as its own ayah (per-verse granularity retained)
        i = j + 1
    return allm


def _mk(verse, res, words, cross) -> VerifiedMatch:
    w0, w1 = words[res.word_start], words[res.word_end]
    conf = compute_confidence(res.fuzz_score, res.matched_tokens)
    status = MatchStatus.accepted if conf >= 0.82 else MatchStatus.review
    return VerifiedMatch(
        surah=verse.surah_name, surah_no=verse.surah_no, ayah=verse.ayah_no,
        ayah_end=verse.ayah_no, verse_id=verse.verse_id,
        canonical_text=verse.text_uthmani,
        canonical_quoted=verse.quoted_uthmani(res.verse_start, res.verse_end),
        confidence=round(conf, 4), status=status,
        start=w0.start, end=w1.end, word_start=res.word_start, word_end=res.word_end,
        transcript_text=" ".join(w.raw for w in words[res.word_start:res.word_end + 1]),
        fuzz_score=round(res.fuzz_score, 2),
        coverage=coverage(res.matched_tokens, len(verse.tokens)),
        is_cross_surah=cross,
    )
