"""
retrieval.py  —  STAGE B : Candidate generation (seed-and-verify)
=================================================================

DESIGN DECISION (the comparison the task asked for)
---------------------------------------------------
We must NOT brute-force all 6236 verses against every transcript window.
Options considered:

  approach            build    query/window   order-aware   external dep   verdict
  ------------------  -------  -------------   -----------   ------------   ------------------
  brute-force ratio   none     6236 ratios     n/a           none           too slow (rejected)
  Whoosh              disk     ~ms             via phrase    Whoosh         heavy, disk I/O, slow build
  SQLite FTS5         disk     ~ms             phrase("..")  stdlib         good, but file-locking flaky
                                                                            on network mounts; per-window
                                                                            SQL round-trips add up
  BM25 (rank_bm25)    ~0.3 s   ~3.3 ms         bag-of-words  rank_bm25      ranks well, but 3.3ms x ~2300
                      (in RAM)                                              windows x 1214 eps ~= 2.6 h
  INVERTED BIGRAM     ~0.2 s   ~0.5 us         exact phrase  none (stdlib)  ~1 ms / WHOLE episode  <-- WIN
  index (chosen)      (in RAM)

=> CHOSEN: an in-RAM inverted index of verse word-BIGRAMS. A bigram shared by
   transcript and verse is a strong, position-bearing seed. Generation is a few
   dict lookups per episode (1000x faster than BM25), and it returns the anchor
   POSITION for free, which Stage C needs. Common bigrams (df > bigram_df_cap)
   are skipped so frequent clauses don't explode the candidate set.

   BM25 is still built (QuranIndex.bm25) and exposed for the product's *search*
   feature and as a `backend: bm25` fallback, but the enrichment hot path uses
   bigrams.

OUTPUT: a list of RawMatch (schema.py) — one per (verse, located span) that
clears `min_partial_score`. De-duplication / cross-surah / multiverse decisions
happen later (enrich.py, multiverse.py).
"""
from __future__ import annotations

from collections import defaultdict

from .config import Config
from .quran_index import QuranIndex
from .schema import RawMatch
from .transcript_io import Word
from .verify import seed_extend


def _bigram_seeds(index: QuranIndex, words: list[Word], df_cap: int) -> dict[int, list[int]]:
    """verse_id -> sorted list of transcript positions where a (rare) shared
    bigram anchors that verse."""
    seeds: dict[int, list[int]] = defaultdict(list)
    norm = [w.norm for w in words]
    for i in range(len(norm) - 1):
        for vid in index.bigram_postings(norm[i], norm[i + 1], df_cap):
            seeds[vid].append(i)
    return seeds


def collect_raw_matches(index: QuranIndex, words: list[Word], cfg: Config) -> list[RawMatch]:
    """STAGE B + the localisation half of STAGE C: produce verified-span raw
    candidates for every seeded verse."""
    df_cap = int(cfg.retrieval["bigram_df_cap"])
    min_score = float(cfg.verify["min_partial_score"])
    norm = [w.norm for w in words]

    seeds = _bigram_seeds(index, words, df_cap)
    raw: list[RawMatch] = []

    for vid, positions in seeds.items():
        verse = index.get(vid)
        if verse is None:
            continue
        accepted_spans: list[tuple[int, int]] = []
        for pos in sorted(positions):
            # skip positions already inside a span we accepted for THIS verse
            if any(s <= pos <= e for s, e in accepted_spans):
                continue
            res = seed_extend(verse.tokens, norm, pos)
            if res is None or res.fuzz_score < min_score:
                continue
            accepted_spans.append((res.word_start, res.word_end))
            w0, w1 = words[res.word_start], words[res.word_end]
            raw.append(RawMatch(
                verse_id=vid,
                surah_no=verse.surah_no,
                ayah_no=verse.ayah_no,
                word_start=res.word_start,
                word_end=res.word_end,
                fuzz_score=res.fuzz_score,
                matched_tokens=res.matched_tokens,
                verse_tokens=len(verse.tokens),
                verse_token_start=res.verse_start,
                verse_token_end=res.verse_end,
                start=w0.start,
                end=w1.end,
                transcript_text=" ".join(w.raw for w in words[res.word_start:res.word_end + 1]),
            ))
    return raw
