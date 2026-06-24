"""
Unit + integration tests for the Quran enrichment pipeline.

Run:   python -m pytest tests/ -q     (or)     python tests/test_enrichment.py
The corpus (data/quran/quran.jsonl) must exist:  python -m quran_alignment.cli build
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quran_alignment.normalize import normalize, normalize_tokens          # noqa: E402
from quran_alignment.config import load_config                            # noqa: E402
from quran_alignment.quran_index import QuranIndex                        # noqa: E402
from quran_alignment.verify import seed_extend, compute_confidence        # noqa: E402
from quran_alignment.enrich import Enricher                               # noqa: E402

_CFG = load_config()
_IDX = QuranIndex(_CFG.quran_db_jsonl)
_ENR = Enricher(_CFG)


# ---------- STAGE A : normalization -----------------------------------------
def test_normalize_strips_all_marks():
    assert normalize("مُوسَىٰ") == "موسي"
    assert normalize("هَٰذَا") == "هذا"
    assert normalize("أُوْلَٰٓئِكَ") == "اوليك"
    assert normalize("ٱلصَّلَوٰةَ") == "الصلوه"          # alef-wasla + ta-marbuta-ish
    assert normalize("السَّمَآءِ") == "السما"            # maddah + dropped hamza

def test_normalize_idempotent():
    once = normalize("إِنَّ ٱللَّهَ غَفُورٌ رَّحِيمٌ")
    assert normalize(once) == once

def test_tail_is_substring():
    verse = _IDX.get(180).text_uthmani               # Baqara:173
    assert normalize("ان الله غفور رحيم") in normalize(verse)


# ---------- corpus -----------------------------------------------------------
def test_corpus_size():
    assert len(_IDX) == 6236

def test_quoted_uthmani_exact_fragment():
    v = _IDX.get(180)
    frag = v.quoted_uthmani(21, 24)
    # exact recited fragment, folds back to the plain ASR form, and is a
    # contiguous substring of the full ayah's Uthmani text:
    assert normalize(frag) == "ان الله غفور رحيم"
    assert frag in v.text_uthmani
    assert len(frag.split()) == 4


# ---------- STAGE C : seed-extend localisation ------------------------------
def test_seed_extend_locates_fragment():
    words = normalize_tokens("قال الله تعالى ان الله غفور رحيم ثم يكمل الشرح")
    v = _IDX.get(180)
    res = seed_extend(v.tokens, words, 5)             # bigram 'غفور رحيم' at idx 5
    assert (res.word_start, res.word_end) == (3, 6)
    assert res.fuzz_score >= 99
    assert words[res.word_start:res.word_end + 1] == ["ان", "الله", "غفور", "رحيم"]

def test_confidence_monotonic_in_length():
    assert compute_confidence(100, 2) < compute_confidence(100, 4) <= compute_confidence(100, 6)


# ---------- end-to-end : the brief's example --------------------------------
def _demo_transcript():
    words = "قال الله تعالى ان الله غفور رحيم ثم يكمل الشرح".split()
    return {"surah": "البقرة", "surah_order": 2, "episode": 1, "source_audio": "demo",
            "segments": [{"id": 0, "start": 120.0, "end": 130.0, "text": " ".join(words),
                          "words": [{"word": w, "start": 120.0 + i, "end": 121.0 + i,
                                     "probability": 0.9} for i, w in enumerate(words)]}]}

def test_end_to_end_match_fields():
    res = _ENR.enrich_transcript(_demo_transcript())
    m = next(m for m in res.matches if m.status.value == "accepted")
    assert m.surah == "البقرة" and m.ayah == 173
    assert normalize(m.canonical_quoted) == "ان الله غفور رحيم"
    assert m.canonical_quoted == _IDX.get(180).quoted_uthmani(21, 24)
    assert m.confidence >= 0.82
    assert (m.start, m.end) == (123.0, 127.0)          # timestamps preserved

def test_end_to_end_replacement_text():
    res = _ENR.enrich_transcript(_demo_transcript())
    frag = _IDX.get(180).quoted_uthmani(21, 24)
    expected = f"قال الله تعالى ﴿ {frag} ﴾ [البقرة:173] ثم يكمل الشرح"
    assert res.segments[0].enriched_text == expected


# ---------- cross-surah ------------------------------------------------------
def test_cross_surah_detected():
    """A verse from a DIFFERENT surah than the episode primary must be flagged."""
    words = "وقال تعالى وأعدوا لهم ما استطعتم من قوة وهذا في غير سورتنا".split()
    tr = {"surah": "آل عمران", "surah_order": 3, "episode": 99, "source_audio": "x",
          "segments": [{"id": 0, "start": 0.0, "end": 20.0, "text": " ".join(words),
                        "words": [{"word": w, "start": float(i), "end": float(i) + 1,
                                   "probability": 0.9} for i, w in enumerate(words)]}]}
    res = _ENR.enrich_transcript(tr)
    anfal = [m for m in res.matches if m.surah == "الأنفال"]
    assert anfal and anfal[0].is_cross_surah is True


# ---------- multi-verse ------------------------------------------------------
def test_multiverse_chain():
    """Two consecutive verses recited back-to-back get a shared chain_id."""
    a = _IDX.get(1)   # الفاتحة:1
    b = _IDX.get(2)   # الفاتحة:2
    toks = a.tokens + b.tokens
    tr = {"surah": "الفاتحة", "surah_order": 1, "episode": 1, "source_audio": "x",
          "segments": [{"id": 0, "start": 0.0, "end": float(len(toks)),
                        "text": " ".join(toks),
                        "words": [{"word": w, "start": float(i), "end": float(i) + 1,
                                   "probability": 0.9} for i, w in enumerate(toks)]}]}
    res = _ENR.enrich_transcript(tr)
    chained = [m for m in res.matches if m.chain_id]
    assert len(chained) >= 2
    assert {m.ayah for m in chained} >= {1, 2}


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for fn in fns:
        try:
            fn(); passed += 1; print(f"PASS {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e!r}")
    print(f"\n{passed}/{len(fns)} tests passed")
