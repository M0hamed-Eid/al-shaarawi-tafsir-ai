"""
verify.py  —  STAGE C : Verse verification & precise localisation
=================================================================

Stage B hands us (verse, transcript-bigram-position) seeds: a 2-word anchor that
exists in BOTH the verse and the transcript. Here we recover the EXACT run of
transcript words that the verse covers, and score it.

Why not plain rapidfuzz.partial_ratio?
--------------------------------------
`partial_ratio` always embeds the SHORTER string fully into the longer one. When
the transcript holds only a short *fragment* of a long verse (e.g. the Sheikh
quotes the tail "إن الله غفور رحيم" of a 28-word verse), partial_ratio embeds the
whole transcript window into the verse and reports the entire window as matched —
wrong span, wrong timestamps. We therefore use **seed-and-extend** token
alignment (the standard approach in near-duplicate / bio-sequence matching):

    1. SEED   : the bigram that triggered the candidate is located in the verse.
    2. EXTEND : walk left & right matching tokens (with a small fuzzy budget for
                ASR substitutions) until the verse or the agreement runs out.
    3. SCORE  : rapidfuzz.ratio on the two aligned, equal-ish-length strings
                (symmetric, order-sensitive) -> clean 0..100 score.

This handles all three regimes uniformly: a short fragment, a full verse, and
(via repeated seeding in multiverse.py) consecutive verses.

Comparison of the scorers the task asked about — and our verdict:
  * ratio / Levenshtein   : great on the FINAL equal-length aligned span (used).
  * partial_ratio         : right idea, wrong span on fragments (see above).
  * token_set_ratio       : order-blind -> unsafe for Quran word order (rejected).
  * token_sort_ratio      : order-blind (rejected).
  * SequenceMatcher       : correct but slower; rapidfuzz.ratio is the C++ equiv.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

# A transcript token "equals" a verse token if identical after normalisation, or
# within a tiny edit distance (1 char) / high ratio — absorbs ASR spelling slips.
def tok_match(a: str, b: str) -> bool:
    if a == b:
        return True
    if abs(len(a) - len(b)) <= 1 and Levenshtein.distance(a, b) <= 1:
        return True
    return len(a) >= 4 and len(b) >= 4 and fuzz.ratio(a, b) >= 85.0


@dataclass
class AlignResult:
    word_start: int       # inclusive index into the transcript word stream
    word_end: int         # inclusive
    fuzz_score: float     # rapidfuzz.ratio of aligned strings, [0,100]
    matched_tokens: int   # transcript words covered by the verse span
    verse_start: int      # inclusive token index inside the verse
    verse_end: int        # inclusive


def _extend(T: list[str], V: list[str], ti: int, vi: int, step: int,
            budget: int) -> tuple[int, int]:
    """From aligned pair (ti,vi) walk in direction `step` (+1/-1). Returns the
    farthest (ti,vi) still anchored by a real token match, tolerating up to
    `budget` substitutions (mismatched-but-advanced pairs)."""
    last_ti, last_vi = ti, vi
    mism = 0
    while True:
        ti += step
        vi += step
        if not (0 <= ti < len(T) and 0 <= vi < len(V)):
            break
        if tok_match(T[ti], V[vi]):
            last_ti, last_vi = ti, vi
        else:
            mism += 1
            if mism > budget:
                break
            # treat as substitution and keep going (do not move the anchor)
    return last_ti, last_vi


def seed_extend(verse_tokens: list[str],
                words_norm: list[str],
                t_pos: int,
                max_mismatch_base: int = 2) -> Optional[AlignResult]:
    """Align `verse_tokens` to the transcript using the bigram seed that starts
    at transcript index `t_pos` (so words_norm[t_pos:t_pos+2] is the bigram)."""
    T, V = words_norm, verse_tokens
    if t_pos + 1 >= len(T) or len(V) < 2:
        return None
    w1, w2 = T[t_pos], T[t_pos + 1]

    best: Optional[AlignResult] = None
    # the bigram may occur at several places in the verse; try each seed
    for k in range(len(V) - 1):
        if V[k] != w1 or V[k + 1] != w2:
            continue
        budget = max_mismatch_base + len(V) // 6
        # extend right from the END of the seed (t_pos+1, k+1)
        r_ti, r_vi = _extend(T, V, t_pos + 1, k + 1, +1, budget)
        # extend left from the START of the seed (t_pos, k)
        l_ti, l_vi = _extend(T, V, t_pos, k, -1, budget)
        w_start, w_end = l_ti, r_ti
        v_start, v_end = l_vi, r_vi
        if w_end < w_start:
            continue
        t_str = " ".join(T[w_start:w_end + 1])
        v_str = " ".join(V[v_start:v_end + 1])
        score = float(fuzz.ratio(t_str, v_str))
        cand = AlignResult(w_start, w_end, score, w_end - w_start + 1, v_start, v_end)
        if best is None or (cand.matched_tokens, cand.fuzz_score) > (best.matched_tokens, best.fuzz_score):
            best = cand
    return best


# ---- confidence -------------------------------------------------------------
def length_factor(matched_tokens: int) -> float:
    """Evidence weight from match length. 1->0.40 2->0.66 3->0.85 4->0.96 5+->1.0."""
    if matched_tokens >= 5:
        return 1.0
    return {1: 0.40, 2: 0.66, 3: 0.85, 4: 0.96}.get(matched_tokens, 0.40)


def compute_confidence(fuzz_score: float, matched_tokens: int) -> float:
    return round((fuzz_score / 100.0) * length_factor(matched_tokens), 4)


def coverage(matched_tokens: int, verse_tokens: int) -> float:
    if verse_tokens <= 0:
        return 0.0
    return round(min(1.0, matched_tokens / verse_tokens), 4)
