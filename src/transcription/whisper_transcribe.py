"""
whisper_transcribe.py
---------------------
Phase 3 — Whisper Transcription

Reads metadata.json + extraction manifest, transcribes WAV files using
faster-whisper, and writes JSON transcripts to data/transcripts/<surah>/.

Features:
  - Skips already-transcribed episodes (resumable)
  - Manifest-based crash recovery
  - Batch processing with configurable size
  - Progress bar per batch
  - Full logging to logs/transcription.log
  - Device-agnostic (CPU now, GPU later — change config in transcriber.py)

Usage:
    python src/transcription/whisper_transcribe.py
    python src/transcription/whisper_transcribe.py --batch-size 20
    python src/transcription/whisper_transcribe.py --dry-run
    python src/transcription/whisper_transcribe.py --limit 10 --dry-run
"""

import argparse
import json
import logging
import sys
from datetime import datetime, UTC
from pathlib import Path

from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR       = Path(__file__).resolve().parents[2]
METADATA_FILE  = ROOT_DIR / "data" / "metadata" / "metadata.json"
AUDIO_DIR      = ROOT_DIR / "data" / "audio"
TRANSCRIPTS_DIR = ROOT_DIR / "data" / "transcripts"
MANIFEST_FILE  = ROOT_DIR / "data" / "processed" / "transcription_manifest.json"
LOG_FILE       = ROOT_DIR / "logs" / "transcription.log"

# Add src/ so relative imports work when running as script
sys.path.insert(0, str(ROOT_DIR / "src"))
from ingestion.process_manifest import ProcessManifest

# Local imports (same package)
sys.path.insert(0, str(Path(__file__).parent))
from transcriber import Transcriber, TRANSCRIPTION_CONFIG
from transcript_schema import EpisodeTranscript


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("transcription")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_metadata(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["videos"]


def build_transcript_path(record: dict) -> Path:
    """Mirror the audio path structure: data/transcripts/<surah>/ep_NNN.json"""
    # Derive the WAV filename from the same logic used in extract_audio.py
    # We reconstruct the stem from the audio path in the manifest if available,
    # otherwise build it from record fields.
    surah_dir = TRANSCRIPTS_DIR / record["surah"]
    episode_num = f"{record['episode']:03d}"

    filename_stem = Path(record["filename"]).stem
    if "الجزء الأول" in filename_stem:
        suffix = "_part1"
    elif "الجزء الثاني" in filename_stem:
        suffix = "_part2"
    elif "الجزء الثالث" in filename_stem:
        suffix = "_part3"
    elif "الجزء الرابع" in filename_stem:
        suffix = "_part4"
    else:
        suffix = ""

    return surah_dir / f"ep_{episode_num}{suffix}.json"


def build_audio_path(record: dict) -> Path:
    """Reconstruct the WAV path produced by Phase 2."""
    surah_dir = AUDIO_DIR / record["surah"]
    episode_num = f"{record['episode']:03d}"

    filename_stem = Path(record["filename"]).stem
    if "الجزء الأول" in filename_stem:
        suffix = "_part1"
    elif "الجزء الثاني" in filename_stem:
        suffix = "_part2"
    elif "الجزء الثالث" in filename_stem:
        suffix = "_part3"
    elif "الجزء الرابع" in filename_stem:
        suffix = "_part4"
    else:
        suffix = ""

    return surah_dir / f"ep_{episode_num}{suffix}.wav"


def print_report(manifest: ProcessManifest, logger: logging.Logger) -> None:
    counts = manifest.counts
    logger.info("─" * 50)
    logger.info("Transcription Report")
    logger.info("  Total   : %d", counts["total"])
    logger.info("  Done    : %d", counts["done"])
    logger.info("  Failed  : %d", counts["failed"])
    logger.info("  Pending : %d", counts["pending"])
    logger.info("─" * 50)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_transcription(
    batch_size: int,
    dry_run: bool,
    limit: int | None,
) -> None:
    logger = setup_logging()
    logger.info("=" * 50)
    logger.info("Phase 3 — Transcription started at %s", datetime.now(UTC).isoformat())
    logger.info(
        "Model: %s | Device: %s | Dry run: %s",
        TRANSCRIPTION_CONFIG["model_size"],
        TRANSCRIPTION_CONFIG["device"],
        dry_run,
    )

    # Load all video metadata
    videos = load_metadata(METADATA_FILE)
    logger.info("Loaded %d video records", len(videos))

    # Initialise manifest
    manifest = ProcessManifest(MANIFEST_FILE, phase="transcription")

    # Register all files; mark as done if transcript already exists on disk
    for record in videos:
        transcript_path = build_transcript_path(record)
        manifest.register(record["filename"], str(transcript_path))

        if transcript_path.exists() and not manifest.is_done(record["filename"]):
            manifest.mark_done(record["filename"])

    manifest.save()

    # Determine what still needs processing
    pending = set(manifest.pending_files())
    to_process = [r for r in videos if r["filename"] in pending]
    if limit:
        to_process = to_process[:limit]

    logger.info(
        "To transcribe: %d  |  Already done: %d",
        len(to_process), len(videos) - len(to_process),
    )

    if not to_process:
        logger.info("Nothing to do — all files already transcribed.")
        print_report(manifest, logger)
        return

    if dry_run:
        logger.info("[DRY RUN] Would transcribe %d files:", len(to_process))
        for r in to_process[:20]:
            audio = build_audio_path(r)
            logger.info("  %s → %s", audio.name, build_transcript_path(r))
        if len(to_process) > 20:
            logger.info("  ... and %d more", len(to_process) - 20)
        return

    # Load the model once — keep it in memory for the entire run
    transcriber = Transcriber(TRANSCRIPTION_CONFIG)
    transcriber.load_model()

    done_count = 0
    fail_count = 0

    try:
        for batch_start in range(0, len(to_process), batch_size):
            batch = to_process[batch_start : batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(to_process) + batch_size - 1) // batch_size

            logger.info(
                "Batch %d / %d  (%d files)",
                batch_num, total_batches, len(batch),
            )

            for record in tqdm(
                batch,
                desc=f"Batch {batch_num}",
                unit="ep",
                dynamic_ncols=True,
                leave=False,
            ):
                filename = record["filename"]
                audio_path = build_audio_path(record)
                transcript_path = build_transcript_path(record)

                if not audio_path.exists():
                    manifest.mark_failed(filename, f"WAV not found: {audio_path}")
                    fail_count += 1
                    logger.error("WAV missing: %s", audio_path)
                    continue

                try:
                    transcript = transcriber.transcribe(audio_path, record)
                    transcript.save(transcript_path)
                    manifest.mark_done(filename)
                    done_count += 1
                    logger.debug("✓ %s (%d segments)", filename, len(transcript.segments))

                except Exception as exc:  # noqa: BLE001
                    manifest.mark_failed(filename, str(exc))
                    fail_count += 1
                    logger.error("✗ %s — %s", filename, exc)

            # Crash-safe: save manifest after every batch
            manifest.save()
            logger.info(
                "Batch %d complete — done: %d, failed: %d",
                batch_num, done_count, fail_count,
            )

    finally:
        # Always unload model cleanly, even on Ctrl+C
        transcriber.unload_model()

    print_report(manifest, logger)
    logger.info("Phase 3 complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Transcribe Al-Shaarawi Tafsir audio using Whisper"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Files per batch before saving manifest (default: 50)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without running Whisper"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N pending files"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_transcription(
        batch_size=args.batch_size,
        dry_run=args.dry_run,
        limit=args.limit,
    )