"""
decide.py  —  STAGE E : cross-surah decision, de-duplication, confidence
========================================================================

Input : RawMatch list (every seeded verse that located a span, any surah).
Output: VerifiedMatch list — overlap-resolved, confidence-scored, status-tagged.

KEY PRINCIPLES
--------------
* GLOBAL search: candidates come from all 114 surahs. The episode's "primary"
  surah is used ONLY as a gentle tie-breaker (primary_surah_boost) and to break
  ambiguity — never as a hard filter (Al-Shaarawi quotes across surahs).
* AMBIGUITY GUARD: many Quranic clauses recur verbatim (e.g.
  "إن الله غفور رحيم", "فبأي آلاء ربكما تكذبان"). If a DIFFERENT verse aligns to
  the same span within `ambiguity_delta` fuzz points, the attribution is
  uncertain: we penalise confidence (so it lands in 'review') unless one
  candidate is the primary surah, which we then prefer.
* FALSE-POSITIVE PREVENTION: min_partial_score (Stage B), length_factor
  (short coincidences can't score high), min_standalone_tokens (a lone 2-word
  hit is never auto-accepted), and disjoint greedy selection.
"""
from __future__ import annotations

from .config import Config
from .quran_index import QuranIndex
from .schema import RawMatch, VerifiedMatch, MatchStatus
from .verify import compute_confidence, coverage


def _overlap(a: RawMatch, b: RawMatch) -> int:
    return min(a.word_end, b.word_end) - max(a.word_start, b.word_start) + 1


def _same_span(a: RawMatch, b: RawMatch) -> bool:
    """True if a and b cover essentially the same transcript span (Jaccard>0.6)."""
    inter = max(0, _overlap(a, b))
    if inter <= 0:
        return False
    union = (a.word_end - a.word_start + 1) + (b.word_end - b.word_start + 1) - inter
    return union > 0 and inter / union >= 0.6


def decide(raw: list[RawMatch], index: QuranIndex, cfg: Config,
           primary_surah_no: int) -> list[VerifiedMatch]:
    v = cfg.verify
    accept_t = float(v["accept_threshold"])
    review_t = float(v["review_threshold"])
    boost = float(v["primary_surah_boost"])
    amb_delta = float(v["ambiguity_delta"])
    amb_pen = float(v["ambiguity_penalty"])
    min_standalone = int(v["min_standalone_tokens"])

    # base confidence (+ primary-surah boost) for every raw candidate
    scored: list[tuple[float, RawMatch]] = []
    for m in raw:
        conf = compute_confidence(m.fuzz_score, m.matched_tokens)
        if m.surah_no == primary_surah_no:
            conf = min(1.0, conf + boost)
        scored.append((conf, m))

    # greedy disjoint selection, best confidence first
    scored.sort(key=lambda t: (t[0], t[1].matched_tokens, t[1].fuzz_score), reverse=True)
    chosen: list[tuple[float, RawMatch]] = []
    for conf, m in scored:
        if any(_overlap(m, c[1]) > 0 for c in chosen):
            continue
        chosen.append((conf, m))

    # ambiguity guard: compare each chosen span against ALL raw competitors
    out: list[VerifiedMatch] = []
    for conf, m in chosen:
        competitors = [
            r for r in raw
            if r.verse_id != m.verse_id and _same_span(r, m)
            and abs(r.fuzz_score - m.fuzz_score) <= amb_delta
        ]
        ambiguous = len(competitors) > 0
        if ambiguous:
            primary_here = (m.surah_no == primary_surah_no)
            competitor_primary = any(c.surah_no == primary_surah_no for c in competitors)
            # penalise unless WE are the unique primary-surah candidate
            if not (primary_here and not competitor_primary):
                conf = max(0.0, conf - amb_pen)

        # status from thresholds
        if conf >= accept_t:
            status = MatchStatus.accepted
        elif conf >= review_t:
            status = MatchStatus.review
        else:
            status = MatchStatus.rejected
        # a lone short hit can never auto-accept
        if m.matched_tokens < min_standalone and status == MatchStatus.accepted:
            status = MatchStatus.review

        verse = index.get(m.verse_id)
        out.append(VerifiedMatch(
            surah=verse.surah_name,
            surah_no=verse.surah_no,
            ayah=verse.ayah_no,
            ayah_end=verse.ayah_no,
            verse_id=verse.verse_id,
            canonical_text=verse.text_uthmani,
            canonical_quoted=verse.quoted_uthmani(m.verse_token_start, m.verse_token_end),
            confidence=round(conf, 4),
            status=status,
            start=m.start,
            end=m.end,
            word_start=m.word_start,
            word_end=m.word_end,
            transcript_text=m.transcript_text,
            fuzz_score=round(m.fuzz_score, 2),
            coverage=coverage(m.matched_tokens, m.verse_tokens),
            is_cross_surah=(verse.surah_no != primary_surah_no),
        ))
    out.sort(key=lambda x: x.word_start)
    return out
