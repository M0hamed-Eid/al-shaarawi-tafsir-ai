# Al-Shaarawi Tafsir AI

An AI-powered Arabic Quranic exegesis archive built over the complete video collection of **Sheikh Muhammad Mitwalli Al-Shaarawi** — 1,214 episodes of Tafsir, fully transcribed, indexed, and searchable.

---

## What This Project Does

This system processes Sheikh Al-Shaarawi's complete Tafsir video archive through a multi-phase pipeline:

| Phase | What It Does | Status |
|-------|-------------|--------|
| **1 — Metadata Indexing** | Parses Arabic filenames, extracts surah/episode info, sorts by Quran order | ✅ Done |
| **2 — Audio Extraction** | Extracts 16kHz mono WAV audio from each video using FFmpeg | ✅ Done |
| **3 — Transcription** | Transcribes Arabic speech to text using Whisper (faster-whisper) | ✅ Done |
| **4 — Verse Alignment** | Aligns transcript segments to specific Quranic ayahs | 🔜 Next |
| **5–7 — RAG Pipeline** | Chunking, embeddings, and retrieval system | 🔜 Planned |
| **8 — Web Application** | Searchable web interface for the full archive | 🔜 Planned |

---

## Project Structure

```
al-shaarawi-tafsir-ai/
├── data/
│   ├── raw_videos/          # Original video files (not tracked in git)
│   ├── audio/               # Extracted WAV files, organized by surah
│   ├── transcripts/         # JSON transcripts, organized by surah
│   ├── metadata/
│   │   ├── metadata.json    # Full index of all 1,214 videos
│   │   ├── statistics.json  # Episode count per surah
│   │   └── surah_order.json # Canonical Quran surah ordering
│   └── processed/
│       ├── extraction_manifest.json    # Phase 2 progress tracker
│       └── transcription_manifest.json # Phase 3 progress tracker
├── src/
│   ├── ingestion/
│   │   ├── scan_videos.py       # Arabic filename parser
│   │   ├── metadata_builder.py  # Phase 1 orchestrator
│   │   └── process_manifest.py  # Resumable processing tracker (shared)
│   ├── transcription/
│   │   ├── extract_audio.py     # Phase 2 orchestrator
│   │   ├── transcriber.py       # faster-whisper wrapper
│   │   ├── transcript_schema.py # Transcript dataclasses
│   │   └── whisper_transcribe.py # Phase 3 orchestrator
│   ├── quran_alignment/         # Phase 4 (upcoming)
│   ├── rag/                     # Phase 5–7 (upcoming)
│   └── api/                     # Phase 8 (upcoming)
├── logs/
│   ├── extraction.log
│   └── transcription.log
├── pyproject.toml
└── README.md
```

---

## Requirements

- **Python 3.14+**
- **FFmpeg** installed and available on your PATH
- **uv** (recommended package manager)

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Install FFmpeg

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows — download from https://ffmpeg.org/download.html
```

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/al-shaarawi-tafsir-ai.git
cd al-shaarawi-tafsir-ai

# 2. Install dependencies
uv sync

# 3. Place your videos
# Copy all video files into:  data/raw_videos/
```

---

## Running the Pipeline

Each phase is independent. Run them in order.

### Phase 1 — Build Metadata Index

Scans `data/raw_videos/`, parses Arabic filenames, and produces a structured index sorted by Quran order.

```bash
uv run python src/ingestion/metadata_builder.py
```

**Output:**
- `data/metadata/metadata.json` — full index of all videos
- `data/metadata/statistics.json` — episode count per surah

---

### Phase 2 — Extract Audio

Reads `metadata.json` and extracts 16kHz mono WAV audio from every video.

```bash
# Standard run
uv run python src/transcription/extract_audio.py

# Preview what will run (no FFmpeg calls)
uv run python src/transcription/extract_audio.py --dry-run

# Process only first 10 files (useful for testing)
uv run python src/transcription/extract_audio.py --limit 10

# Control batch size (default: 100)
uv run python src/transcription/extract_audio.py --batch-size 50
```

**Output:** WAV files at `data/audio/<surah_name>/ep_NNN.wav`

> If interrupted, just re-run — already extracted files are skipped automatically.

---

### Phase 3 — Transcribe Audio

Reads extracted WAVs and produces JSON transcripts using Whisper.

```bash
# Standard run (loads model once, processes all files)
uv run python src/transcription/whisper_transcribe.py

# Preview only
uv run python src/transcription/whisper_transcribe.py --dry-run

# Process only first 5 files (test run)
uv run python src/transcription/whisper_transcribe.py --limit 5

# Control batch size (default: 50)
uv run python src/transcription/whisper_transcribe.py --batch-size 20
```

**Output:** JSON transcripts at `data/transcripts/<surah_name>/ep_NNN.json`

> Resume-safe: if the process crashes or is stopped, re-running picks up from where it left off.

---

## GPU Acceleration (Faster Transcription)

By default, the project runs on **CPU** using the `medium` Whisper model. On a GPU machine, you can switch to the `large-v3` model for significantly better accuracy and speed.

Open `src/transcription/transcriber.py` and change **these 3 lines** inside `TRANSCRIPTION_CONFIG`:

```python
# ── BEFORE (CPU) ──────────────────────────────────────────
"model_size": "medium",
"device":      "cpu",
"compute_type": "int8",

# ── AFTER (GPU) ───────────────────────────────────────────
"model_size":   "large-v3",
"device":       "cuda",
"compute_type": "float16",
```

That's it. No other changes needed. The rest of the pipeline is already device-agnostic.

> **Note:** GPU mode requires `nvidia-driver`, `CUDA`, and installing `faster-whisper` with CUDA support. Recommended: 8+ GB VRAM for `large-v3`.

---

## Transcript Format

Each episode is saved as a JSON file with this structure:

```json
{
  "schema_version": "1.0",
  "surah": "البقرة",
  "surah_order": 2,
  "episode": 14,
  "source_audio": "data/audio/البقرة/ep_014.wav",
  "transcribed_at": "2026-06-08T12:00:00Z",
  "model": "medium",
  "language": "ar",
  "language_probability": 0.998,
  "duration_seconds": 3240.5,
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 4.82,
      "text": "بسم الله الرحمن الرحيم",
      "avg_logprob": -0.21,
      "no_speech_prob": 0.01,
      "words": [
        { "word": "بسم", "start": 0.0, "end": 0.6, "probability": 0.99 },
        ...
      ]
    }
  ]
}
```

Confidence scores (`avg_logprob`, `no_speech_prob`) and word-level timestamps are included from the start, designed for downstream RAG filtering and Phase 4 verse alignment.

---

## Key Design Decisions

**Resumable by default** — every phase uses a manifest file to track progress. You can stop and restart at any time without reprocessing completed files.

**Config-driven** — Whisper settings are centralized in one dict in `transcriber.py`. Switching from CPU to GPU is 3 lines.

**Arabic-aware parsing** — the filename parser handles 6 different Arabic naming patterns and a `SPELLING_MAP` for orthographic variants (e.g. `ابراهيم` → `إبراهيم`).

**Built for RAG** — transcript JSON schema embeds confidence scores and word timestamps from Phase 3 to support quality filtering and verse alignment in later phases.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `faster-whisper` | Arabic speech-to-text transcription |
| `tqdm` | Progress bars for batch processing |
| `FFmpeg` | Audio extraction from video files (system dependency) |

---

## License

This project is for educational and research purposes. All Tafsir content belongs to its respective rights holders.