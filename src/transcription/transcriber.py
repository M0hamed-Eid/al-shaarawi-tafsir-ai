"""
transcriber.py
--------------
Wraps faster-whisper with sensible defaults for Arabic Tafsir content.
Designed to be device-agnostic: works on CPU, scales to CUDA/GPU.
"""

import logging
from datetime import datetime, UTC
from pathlib import Path

from faster_whisper import WhisperModel

from transcript_schema import (
    EpisodeTranscript,
    TranscriptSegment,
    WordTimestamp,
)

logger = logging.getLogger("transcriber")


# ── Configuration ─────────────────────────────────────────────────────────────

TRANSCRIPTION_CONFIG = {
    # Model: "tiny", "base", "small", "medium", "large-v3"
    # On CPU: use "medium". On GPU with 8+ GB VRAM: use "large-v3"
    "model_size": "medium",

    # Device: "cpu" or "cuda". Set to "cuda" when moving to GPU machine.
    "device": "cpu",

    # Compute type:
    #   CPU:  "int8"           (fast, small memory footprint)
    #   GPU:  "float16"        (fast, accurate)
    #         "int8_float16"   (faster, minor accuracy tradeoff)
    "compute_type": "int8",

    # Whisper transcription parameters
    "language": "ar",
    "beam_size": 5,
    "vad_filter": True,             # Silero VAD — reduces hallucinations on silence
    "vad_parameters": {
        "min_silence_duration_ms": 500,
        "speech_pad_ms": 400,
    },
    "word_timestamps": True,        # needed for Phase 4 verse alignment
    "condition_on_previous_text": True,  # helps with Quranic vocabulary consistency

    # Quality thresholds for flagging low-confidence output
    "min_avg_logprob": -1.0,
    "max_no_speech_prob": 0.6,
}


class Transcriber:
    """
    Loads the Whisper model once, transcribes multiple episodes.
    Call transcribe() per episode — model stays loaded between calls.
    """

    def __init__(self, config: dict = TRANSCRIPTION_CONFIG):
        self.config = config
        self._model: WhisperModel | None = None

    def load_model(self) -> None:
        """Load the model into memory. Call once before batch processing."""
        logger.info(
            "Loading Whisper model: %s on %s (%s)",
            self.config["model_size"],
            self.config["device"],
            self.config["compute_type"],
        )
        self._model = WhisperModel(
            self.config["model_size"],
            device=self.config["device"],
            compute_type=self.config["compute_type"],
        )
        logger.info("Model loaded.")

    def transcribe(
        self,
        audio_path: Path,
        metadata: dict,
    ) -> EpisodeTranscript:
        """
        Transcribe one audio file.

        Args:
            audio_path: Path to the WAV file.
            metadata: Dict with surah, surah_order, episode fields.

        Returns:
            EpisodeTranscript with all segments and timestamps.
        """
        if self._model is None:
            raise RuntimeError("Call load_model() before transcribe().")

        logger.debug("Transcribing: %s", audio_path.name)

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=self.config["language"],
            beam_size=self.config["beam_size"],
            vad_filter=self.config["vad_filter"],
            vad_parameters=self.config["vad_parameters"],
            word_timestamps=self.config["word_timestamps"],
            condition_on_previous_text=self.config["condition_on_previous_text"],
        )

        # Materialise the generator — segments stream from the model
        segments = []
        for i, seg in enumerate(segments_iter):
            words = []
            if self.config["word_timestamps"] and seg.words:
                words = [
                    WordTimestamp(
                        word=w.word,
                        start=round(w.start, 3),
                        end=round(w.end, 3),
                        probability=round(w.probability, 4),
                    )
                    for w in seg.words
                ]

            ts = TranscriptSegment(
                id=i,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text.strip(),
                avg_logprob=round(seg.avg_logprob, 4),
                no_speech_prob=round(seg.no_speech_prob, 4),
                words=words,
            )
            segments.append(ts)

            # Log low-confidence segments for review
            if not ts.is_reliable(
                self.config["min_avg_logprob"],
                self.config["max_no_speech_prob"],
            ):
                logger.warning(
                    "Low confidence segment [%.1fs-%.1fs]: logprob=%.2f, no_speech=%.2f",
                    ts.start, ts.end, ts.avg_logprob, ts.no_speech_prob,
                )

        transcript = EpisodeTranscript(
            schema_version="1.0",
            surah=metadata["surah"],
            surah_order=metadata["surah_order"],
            episode=metadata["episode"],
            source_audio=str(audio_path),
            transcribed_at=datetime.now(UTC).isoformat() + "Z",
            model=self.config["model_size"],
            language=info.language,
            language_probability=round(info.language_probability, 4),
            duration_seconds=round(info.duration, 2),
            segments=segments,
        )

        logger.debug(
            "Done: %d segments, %.1fs, lang_prob=%.2f",
            len(segments), info.duration, info.language_probability,
        )
        return transcript

    def unload_model(self) -> None:
        """Release model from memory."""
        self._model = None
        logger.info("Model unloaded.")