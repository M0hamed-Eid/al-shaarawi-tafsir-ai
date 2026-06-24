"""
quran_index.py  —  Quran corpus store + STAGE B retrieval backend
=================================================================

DATA SOURCE
-----------
Tanzil "Uthmani" text (the official, scholar-verified mushaf with full
tashkeel). It ships inside the `pyquran` package as `quran-uthmani.xml`; the
build step copies it into data/quran/ so the project is reproducible offline.

WHY TANZIL  (vs Quran.com / alquran.cloud / fawazahmed0 / others)
-----------------------------------------------------------------
* Tanzil is the UPSTREAM source most other datasets are derived from. Using it
  directly avoids a second-hand transcription layer and its diff risk.
* It provides a true Uthmani edition WITH tashkeel and a documented, stable
  numbering of all 6236 ayat — exactly what we must emit as canonical_text.
* Plain XML/TXT, offline, no API key / rate limit (Quran.com's API is excellent
  for apps but adds a network dependency; its bulk text traces back to Tanzil).
=> Tanzil Uthmani is the canonical store. The simplified/clean rasm is derived
   on the fly by our normalisation layer, so we keep ONE source of truth.

STORAGE  (why JSONL, not SQLite, by default)
--------------------------------------------
The corpus is 6236 very short documents. The whole retrieval index lives in
RAM (rank-bm25), so a server-grade DB buys nothing on the hot path. We persist
the corpus as newline-delimited JSON — robust on ANY filesystem (incl. network
mounts where SQLite file-locking misbehaves) and trivially diff-able.

An OPTIONAL SQLite/FTS5 store is provided (`build_sqlite_fts`) for callers who
want ad-hoc SQL access; it is NOT on the matching hot path.
"""
from __future__ import annotations

import json
import sqlite3
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from shutil import copyfile
from typing import Optional

from rank_bm25 import BM25Okapi

from .normalize import normalize, normalize_cached


# --------------------------------------------------------------------------- #
#  Build                                                                       #
# --------------------------------------------------------------------------- #
def locate_tanzil_xml(preferred: Path) -> Path:
    """Path to the Tanzil Uthmani XML, falling back to the copy bundled with
    pyquran if `preferred` is missing."""
    if preferred.exists():
        return preferred
    try:
        import pyquran
        cand = Path(pyquran.__file__).parent / "QuranCorpus" / "quran-uthmani.xml"
        if cand.exists():
            return cand
    except Exception:
        pass
    raise FileNotFoundError(
        f"Tanzil XML not found at {preferred} and pyquran is unavailable. "
        f"`pip install pyquran` or place quran-uthmani.xml there."
    )


def _iter_verses_from_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    vid = 0
    for sura in root.findall("sura"):
        surah_no = int(sura.attrib["index"])
        surah_name = sura.attrib["name"]
        for aya in sura.findall("aya"):
            ayah_no = int(aya.attrib["index"])
            text = aya.attrib["text"]
            vid += 1
            # Map each surviving normalised token back to its Uthmani token, so
            # we can later quote the EXACT recited fragment with tashkeel even
            # when a token splits/drops under normalisation.
            uth_tokens = text.split()
            norm_parts = []
            uth_align = []
            for ui, ut in enumerate(uth_tokens):
                for sub in normalize(ut).split():
                    norm_parts.append(sub)
                    uth_align.append(ui)
            yield {
                "verse_id": vid,
                "surah_no": surah_no,
                "surah_name": surah_name,
                "ayah_no": ayah_no,
                "text_uthmani": text,
                "text_norm": " ".join(norm_parts),
                "uth_align": uth_align,
            }


def build_corpus(xml_path: Path, jsonl_path: Path,
                 *, copy_xml_to: Optional[Path] = None) -> int:
    """Parse Tanzil XML -> write the JSONL corpus. Returns verse count (6236)."""
    xml_path = locate_tanzil_xml(xml_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    if copy_xml_to and not copy_xml_to.exists():
        copy_xml_to.parent.mkdir(parents=True, exist_ok=True)
        copy_xml_to.write_bytes(Path(xml_path).read_bytes())

    rows = list(_iter_verses_from_xml(xml_path))
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def build_sqlite_fts(xml_path: Path, db_path: Path) -> int:
    """OPTIONAL: build a SQLite + FTS5 store for ad-hoc SQL/full-text queries.
    Built in a local temp dir then copied (some mounts reject direct sqlite
    writes). Not used by the matching pipeline."""
    xml_path = locate_tanzil_xml(xml_path)
    tmp = Path(tempfile.gettempdir()) / f"{db_path.stem}.build.sqlite3"
    if tmp.exists():
        tmp.unlink()
    con = sqlite3.connect(tmp)
    con.execute("""CREATE TABLE verses(
        verse_id INTEGER PRIMARY KEY, surah_no INT, surah_name TEXT,
        ayah_no INT, text_uthmani TEXT, text_norm TEXT)""")
    con.execute("CREATE VIRTUAL TABLE verses_fts USING fts5(text_norm, content='')")
    rows = list(_iter_verses_from_xml(xml_path))
    con.executemany("INSERT INTO verses VALUES(?,?,?,?,?,?)",
                    [(r["verse_id"], r["surah_no"], r["surah_name"], r["ayah_no"],
                      r["text_uthmani"], r["text_norm"]) for r in rows])
    con.executemany("INSERT INTO verses_fts(rowid, text_norm) VALUES(?,?)",
                    [(r["verse_id"], r["text_norm"]) for r in rows])
    con.commit(); con.close()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    copyfile(tmp, db_path)
    return len(rows)


# --------------------------------------------------------------------------- #
#  Load / query                                                               #
# --------------------------------------------------------------------------- #
class Verse:
    __slots__ = ("verse_id", "surah_no", "surah_name", "ayah_no",
                 "text_uthmani", "text_norm", "tokens", "uth_tokens", "uth_align")

    def __init__(self, d: dict):
        self.verse_id = d["verse_id"]
        self.surah_no = d["surah_no"]
        self.surah_name = d["surah_name"]
        self.ayah_no = d["ayah_no"]
        self.text_uthmani = d["text_uthmani"]
        self.text_norm = d["text_norm"]
        self.tokens = self.text_norm.split()
        self.uth_tokens = d["text_uthmani"].split()
        # alignment from norm-token index -> uthmani-token index (fallback: identity)
        self.uth_align = d.get("uth_align") or list(range(len(self.uth_tokens)))

    def __repr__(self):
        return f"<Verse {self.surah_name}:{self.ayah_no} ({len(self.tokens)}t)>"

    def quoted_uthmani(self, vstart: int, vend: int) -> str:
        """Exact Uthmani (with tashkeel) for the normalised-token span
        [vstart, vend] of this verse. Returns the full verse if span invalid."""
        if not (0 <= vstart <= vend < len(self.uth_align)):
            return self.text_uthmani
        a = self.uth_align[vstart]
        b = self.uth_align[vend]
        if not (0 <= a <= b < len(self.uth_tokens)):
            return self.text_uthmani
        return " ".join(self.uth_tokens[a:b + 1])


class QuranIndex:
    """Loads the JSONL corpus and exposes BM25 candidate retrieval (Stage B)
    plus verse-navigation helpers used by the multi-verse chainer (Stage D)."""

    def __init__(self, jsonl_path: Path):
        self.verses: list[Verse] = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.verses.append(Verse(json.loads(line)))
        if not self.verses:
            raise RuntimeError(f"No verses in {jsonl_path}; run build_quran_db.py.")
        self._by_id = {v.verse_id: v for v in self.verses}
        # BM25 over normalised tokens. Empty docs get a sentinel so avgdl != 0.
        corpus = [v.tokens if v.tokens else ["∅"] for v in self.verses]
        self.bm25 = BM25Okapi(corpus)

        # Inverted bigram index for FAST candidate generation (Stage B default).
        # bigram (w_i, w_{i+1}) -> sorted list of verse_ids that contain it.
        from collections import defaultdict
        big = defaultdict(list)
        for v in self.verses:
            t = v.tokens
            for i in range(len(t) - 1):
                big[(t[i], t[i + 1])].append(v.verse_id)
        self.bigram_index = {k: tuple(vs) for k, vs in big.items()}

    # ---- lookups ------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.verses)

    def get(self, verse_id: int) -> Optional[Verse]:
        return self._by_id.get(verse_id)

    def next_verse(self, verse_id: int) -> Optional[Verse]:
        """Verse immediately after `verse_id` IF it is in the same surah."""
        cur = self._by_id.get(verse_id)
        nxt = self._by_id.get(verse_id + 1)
        if cur and nxt and nxt.surah_no == cur.surah_no:
            return nxt
        return None

    # ---- STAGE B : candidate retrieval -------------------------------------
    def query(self, tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        """Return up to top_k (verse_index, bm25_score) for a token window.
        verse_index is the 0-based position in self.verses (== verse_id-1)."""
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(i, float(scores[i])) for i in idx if scores[i] > 0.0]

    # ---- STAGE B : bigram candidate generation (fast path) -----------------
    def bigram_postings(self, w1: str, w2: str, df_cap: int):
        """verse_ids containing bigram (w1,w2), or () if absent / too common."""
        p = self.bigram_index.get((w1, w2))
        if p is None or len(p) > df_cap:
            return ()
        return p
