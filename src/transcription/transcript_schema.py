"""
transcript_schema.py
--------------------
Pydantic-style dataclasses for transcription output.
Used by both the transcriber and downstream phases (alignment, chunking).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json
from pathlib import Path


@dataclass
class WordTimestamp:
    word: str
    start: float
    end: float
    probability: float


@dataclass
class TranscriptSegment:
    id: int
    start: float
    end: float
    text: str
    avg_logprob: float
    no_speech_prob: float
    words: list[WordTimestamp] = field(default_factory=list)

    def is_reliable(self, min_logprob: float = -1.0, max_no_speech: float = 0.6) -> bool:
        """Returns True if segment confidence meets thresholds."""
        return (
            self.avg_logprob >= min_logprob
            and self.no_speech_prob <= max_no_speech
        )


@dataclass
class EpisodeTranscript:
    schema_version: str
    surah: str
    surah_order: int
    episode: int
    source_audio: str
    transcribed_at: str
    model: str
    language: str
    language_probability: float
    duration_seconds: float
    segments: list[TranscriptSegment] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "EpisodeTranscript":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        segments = [
            TranscriptSegment(
                **{k: v for k, v in seg.items() if k != "words"},
                words=[WordTimestamp(**w) for w in seg.get("words", [])]
            )
            for seg in data.pop("segments", [])
        ]
        return cls(**data, segments=segments)

    @property
    def full_text(self) -> str:
        """Concatenate all segment texts. Useful for quick inspection."""
        return " ".join(seg.text.strip() for seg in self.segments)

    @property
    def reliable_segments(self) -> list[TranscriptSegment]:
        return [s for s in self.segments if s.is_reliable()]