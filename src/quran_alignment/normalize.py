"""
normalize.py  —  STAGE A : Normalization layer
==============================================

The Uthmani mushaf is fully vocalised (tashkeel, dagger-alef, quranic
annotation signs, ornate hamza seats). Whisper output for Arabic is, in
practice, almost diacritic-free and uses the "plain" letter forms. To compare
the two we project BOTH sides into the same diacritic-free, letter-folded
space, while ALWAYS keeping the original Uthmani text untouched for display.

Folding rules (all togglable via config["normalization"]):
    * Unicode NFC normalisation
    * Tatweel (U+0640) removal
    * Diacritic removal  : harakat, tanwin, shadda, sukun, maddah, combining
                           hamza seats, dagger alef, full Quranic annotation set
    * Alef folding       : U+0622/0623/0625/0671 -> U+0627  (alef)
    * Alef maqsura       : U+0649 -> U+064A             (alef maqsura -> yeh)
    * Ta marbuta         : U+0629 -> U+0647             (ta marbuta -> heh)
    * Hamza folding      : U+0624 -> U+0648 , U+0626 -> U+064A , U+0621 dropped
    * Strip non-Arabic   : keep only Arabic letters (U+0621-U+064A) + spaces

Pure-Python / regex only. Every Arabic codepoint is referenced numerically via
chr() so the source stays ASCII and the ranges can never be silently corrupted.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Iterable

# --- character classes (numeric codepoints only) -----------------------------
TATWEEL = chr(0x0640)

# Combining marks to strip, as (start, end) codepoint ranges:
_DIACRITIC_RANGES = [
    (0x0610, 0x061A),  # Arabic small high signs (sallallahu, etc.)
    (0x064B, 0x065F),  # harakat, tanwin, shadda, sukun, maddah, hamza seats
    (0x0670, 0x0670),  # superscript (dagger) alef
    (0x06D6, 0x06DC),  # Quranic annotation signs
    (0x06DF, 0x06E4),  # small high marks
    (0x06E7, 0x06E8),  # small high marks
    (0x06EA, 0x06ED),  # small low/high marks, sajda
    (0x08D3, 0x08FF),  # Arabic Extended-A combining marks
]
_DIACRITICS_RE = re.compile(
    "[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in _DIACRITIC_RANGES) + "]"
)

# Keep only Arabic letters (U+0621-U+064A) and spaces; everything else -> space.
_NON_ARABIC_RE = re.compile("[^" + chr(0x0621) + "-" + chr(0x064A) + " ]+")
_MULTISPACE_RE = re.compile(r"\s+")

# Letter-folding translation tables (applied AFTER diacritic removal).
# str.translate accepts {codepoint: codepoint|None}.
_ALEF_TABLE = {0x0622: 0x0627, 0x0623: 0x0627, 0x0625: 0x0627, 0x0671: 0x0627}
_MAQSURA_TABLE = {0x0649: 0x064A}
_TA_MARBUTA_TABLE = {0x0629: 0x0647}
_HAMZA_TABLE = {0x0624: 0x0648, 0x0626: 0x064A, 0x0621: None}


def normalize(text: str, opts: dict | None = None) -> str:
    """Fold `text` into the comparison space. `opts` mirrors
    config['normalization']; opts=None enables every rule (production default)."""
    if not text:
        return ""
    o = opts or {}

    def on(key: str) -> bool:
        return o.get(key, True)

    s = unicodedata.normalize("NFC", text)            # 1. canonical unicode
    if on("remove_tatweel"):
        s = s.replace(TATWEEL, "")                    # 2. tatweel
    if on("remove_diacritics"):
        s = _DIACRITICS_RE.sub("", s)                 # 3. diacritics (before folding!)
    if on("normalize_alef"):
        s = s.translate(_ALEF_TABLE)                  # 4. letter folding ...
    if on("normalize_alef_maqsura"):
        s = s.translate(_MAQSURA_TABLE)
    if on("normalize_ta_marbuta"):
        s = s.translate(_TA_MARBUTA_TABLE)
    if on("normalize_hamza"):
        s = s.translate(_HAMZA_TABLE)
    if on("strip_non_arabic"):
        s = _NON_ARABIC_RE.sub(" ", s)                # 5. drop non-letters
    s = _MULTISPACE_RE.sub(" ", s).strip()            # 6. collapse whitespace
    return s


def normalize_tokens(text: str, opts: dict | None = None) -> list[str]:
    """Normalize then split on whitespace, dropping empties."""
    n = normalize(text, opts)
    return n.split() if n else []


@lru_cache(maxsize=100_000)
def normalize_cached(text: str) -> str:
    """Full-strength normalisation with memoisation (used for the static Quran
    corpus, where the same verse text is folded repeatedly)."""
    return normalize(text, None)


def iter_normalized(texts: Iterable[str], opts: dict | None = None) -> list[str]:
    return [normalize(t, opts) for t in texts]
