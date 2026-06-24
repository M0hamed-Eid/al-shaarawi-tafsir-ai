"""
schema.py  —  OUTPUT DESIGN (Pydantic models)
=============================================

Three layers, each a strict superset of the previous, so traceability is never
lost as data flows Raw -> Verified -> Enriched:

    RawMatch        A candidate produced by retrieval + alignment, BEFORE the
                    accept/review/reject decision. Carries the raw fuzzy score
                    and the transcript word-span it aligned to.

    VerifiedMatch   A RawMatch that passed (or was flagged by) the decision
                    logic. Adds canonical Uthmani text, surah name, ayah
                    number(s), final confidence, timestamps and a status.

    EnrichedSegment / EnrichedTranscript
                    A transcript segment / episode augmented with the verified
                    matches and an optional `enriched_text` where each Quranic
                    quotation has been replaced inline with the canonical text.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MatchStatus(str, Enum):
    accepted = "accepted"   # confidence >= accept_threshold
    review = "review"       # review_threshold <= confidence < accept_threshold
    rejected = "rejected"   # below review_threshold (normally dropped)


class RawMatch(BaseModel):
    """Pre-decision candidate. One verse aligned to one transcript word-span."""
    verse_id: int                      # global 1..6236
    surah_no: int
    ayah_no: int
    # transcript word-span (indices into the flattened word stream)
    word_start: int
    word_end: int                      # inclusive
    # raw rapidfuzz partial-alignment score in [0,100]
    fuzz_score: float
    # how many transcript tokens the verse covered
    matched_tokens: int
    verse_tokens: int                  # total tokens in the verse (normalised)
    verse_token_start: int = 0         # aligned span inside the verse (norm idx)
    verse_token_end: int = 0
    # timestamps copied from the word stream
    start: float
    end: float
    transcript_text: str               # raw transcript words for the span

    class Config:
        frozen = False


class VerifiedMatch(BaseModel):
    """Post-decision match with canonical enrichment + traceability."""
    surah: str                         # Arabic surah name, e.g. "البقرة"
    surah_no: int
    ayah: int                          # first ayah of the match
    ayah_end: Optional[int] = None     # last ayah (== ayah for single-verse)
    verse_id: int
    canonical_text: str                # FULL ayah, Uthmani WITH tashkeel
    canonical_quoted: str = ""          # EXACT recited fragment, Uthmani w/ tashkeel
    confidence: float = Field(ge=0.0, le=1.0)
    status: MatchStatus = MatchStatus.accepted
    start: float                       # seconds
    end: float                         # seconds
    word_start: int
    word_end: int
    transcript_text: str               # what the ASR produced for this span
    # provenance / debug
    fuzz_score: float
    coverage: float                    # matched_tokens / verse_tokens
    is_cross_surah: bool = False       # True if surah_no != episode primary surah
    chain_id: Optional[int] = None     # set when part of a multi-verse chain

    @property
    def ayah_label(self) -> str:
        if self.ayah_end and self.ayah_end != self.ayah:
            return f"{self.surah}:{self.ayah}-{self.ayah_end}"
        return f"{self.surah}:{self.ayah}"


class EnrichedSegment(BaseModel):
    id: int
    start: float
    end: float
    text: str                          # original ASR text
    enriched_text: Optional[str] = None
    matches: list[VerifiedMatch] = Field(default_factory=list)


class EnrichedTranscript(BaseModel):
    schema_version: str = "enrich-1.0"
    surah: str                         # episode primary surah (from metadata)
    surah_order: int
    episode: int
    source_transcript: str
    primary_surah_no: int
    total_matches: int = 0
    accepted: int = 0
    review: int = 0
    cross_surah: int = 0
    matches: list[VerifiedMatch] = Field(default_factory=list)
    segments: list[EnrichedSegment] = Field(default_factory=list)

    def recount(self) -> "EnrichedTranscript":
        self.total_matches = len(self.matches)
        self.accepted = sum(m.status == MatchStatus.accepted for m in self.matches)
        self.review = sum(m.status == MatchStatus.review for m in self.matches)
        self.cross_surah = sum(m.is_cross_surah for m in self.matches)
        return self
