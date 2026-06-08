"""
extract_audio.py
----------------
Phase 2 — Audio Extraction

Reads metadata.json, extracts audio from every video using FFmpeg,
and writes 16 kHz mono WAV files to data/audio/<surah>/.

Features:
  - Skips already-extracted files (resumable)
  - Saves a manifest to data/processed/extraction_manifest.json
  - Logs progress + errors to logs/extraction.log
  - Supports --batch-size for memory-safe processing

Usage:
    python src/transcription/extract_audio.py
    python src/transcription/extract_audio.py --batch-size 50
    python src/transcription/extract_audio.py --dry-run
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path

from tqdm import tqdm

# ── Project paths (resolve from this file's location) ────────────────────────

ROOT_DIR     = Path(__file__).resolve().parents[2]
METADATA_FILE = ROOT_DIR / "data" / "metadata" / "metadata.json"
AUDIO_DIR    = ROOT_DIR / "data" / "audio"
MANIFEST_FILE = ROOT_DIR / "data" / "processed" / "extraction_manifest.json"
LOG_FILE     = ROOT_DIR / "logs" / "extraction.log"

# Add src/ to path so we can import process_manifest
sys.path.insert(0, str(ROOT_DIR / "src"))
from ingestion.process_manifest import ProcessManifest

# ── FFmpeg settings ───────────────────────────────────────────────────────────

FFMPEG_CMD_TEMPLATE = [
    "ffmpeg",
    "-y",               # overwrite output without asking
    "-i", "{input}",    # input video path
    "-vn",              # drop video stream
    "-acodec", "pcm_s16le",
    "-ar", "16000",     # 16 kHz sample rate
    "-ac", "1",         # mono
    "{output}",
]


# ── Logging setup ─────────────────────────────────────────────────────────────

def setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("extract_audio")
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")

    # File handler — full detail
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Console handler — INFO and above only
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ── Core helpers ──────────────────────────────────────────────────────────────

def check_ffmpeg() -> None:
    """Raise RuntimeError if FFmpeg is not available on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "FFmpeg not found. Install it and make sure it is on your PATH."
        )


def load_metadata(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["videos"]


def build_output_path(record: dict) -> Path:
    """
    Map a metadata record to its WAV output path.
    Pattern: data/audio/<surah_name>/ep_<NNN>.wav
    Multi-part episodes (same number, different files) get a suffix.
    """
    surah_dir = AUDIO_DIR / record["surah"]
    episode_num = f"{record['episode']:03d}"

    # Derive part suffix from filename if it contains جزء
    filename_stem = Path(record["filename"]).stem
    if "الجزء الأول" in filename_stem or "الجزء_الأول" in filename_stem:
        suffix = "_part1"
    elif "الجزء الثاني" in filename_stem or "الجزء_الثاني" in filename_stem:
        suffix = "_part2"
    elif "الجزء الثالث" in filename_stem:
        suffix = "_part3"
    elif "الجزء الرابع" in filename_stem:
        suffix = "_part4"
    else:
        suffix = ""

    return surah_dir / f"ep_{episode_num}{suffix}.wav"


def extract_single(video_path: str, wav_path: Path, logger: logging.Logger) -> None:
    """
    Run FFmpeg to extract audio from one video file.
    Raises subprocess.CalledProcessError on failure.
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        part.format(input=video_path, output=str(wav_path))
        if "{input}" in part or "{output}" in part else part
        for part in FFMPEG_CMD_TEMPLATE
    ]

    logger.debug("FFmpeg: %s", " ".join(cmd))

    result = subprocess.run(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.PIPE,
    encoding="utf-8",
    errors="replace",
    )

    if result.returncode != 0:
        # Log the last 5 lines of stderr for diagnostics
        stderr_tail = "\n".join(result.stderr.strip().splitlines()[-5:])
        raise subprocess.CalledProcessError(
            result.returncode, cmd, stderr=stderr_tail
        )


def print_report(manifest: ProcessManifest, logger: logging.Logger) -> None:
    counts = manifest.counts
    logger.info("─" * 50)
    logger.info("Extraction Report")
    logger.info("  Total   : %d", counts["total"])
    logger.info("  Done    : %d", counts["done"])
    logger.info("  Failed  : %d", counts["failed"])
    logger.info("  Pending : %d", counts["pending"])
    logger.info("─" * 50)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_extraction(
    batch_size: int,
    dry_run: bool,
    limit: int | None,
    ) -> None:
    logger = setup_logging()
    logger.info("=" * 50)
    logger.info("Phase 2 — Audio Extraction started at %s",
                datetime.now(UTC).isoformat())
    logger.info("Dry run: %s | Batch size: %d", dry_run, batch_size)

    check_ffmpeg()
    logger.info("FFmpeg available ✓")

    # Load metadata
    videos = load_metadata(METADATA_FILE)
    logger.info("Loaded %d video records from metadata.json", len(videos))

    # Initialise manifest
    manifest = ProcessManifest(MANIFEST_FILE, phase="extraction")

    # Register all files (no-op for already-registered ones)
    for record in videos:
        wav_path = build_output_path(record)
        manifest.register(record["filename"], str(wav_path))

        # If WAV already exists on disk, mark done immediately (handles
        # the case where manifest was deleted but files were not)
        if wav_path.exists() and not manifest.is_done(record["filename"]):
            manifest.mark_done(record["filename"])

    manifest.save()

    # Collect what still needs processing
    pending_filenames = set(manifest.pending_files())
    to_process = [r for r in videos if r["filename"] in pending_filenames]
    if limit:
        to_process = to_process[:limit]

    logger.info("To process : %d  |  Already done : %d",
                len(to_process), len(videos) - len(to_process))

    if not to_process:
        logger.info("Nothing to do — all files already extracted.")
        print_report(manifest, logger)
        return

    # Process in batches
    done_count = 0
    fail_count = 0

    for batch_start in range(0, len(to_process), batch_size):
        batch = to_process[batch_start: batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(to_process) + batch_size - 1) // batch_size
        logger.info("Batch %d / %d  (%d files)", batch_num, total_batches, len(batch))

        for record in tqdm(batch, desc=f"Batch {batch_num}", unit="file",
                           dynamic_ncols=True, leave=False):
            filename = record["filename"]
            video_path = record["path"]
            wav_path = build_output_path(record)

            if dry_run:
                logger.debug("[DRY RUN] Would extract: %s → %s", filename, wav_path)
                continue

            try:
                extract_single(video_path, wav_path, logger)
                manifest.mark_done(filename)
                done_count += 1
                logger.debug("✓ %s", filename)

            except subprocess.CalledProcessError as exc:
                manifest.mark_failed(filename, str(exc.stderr))
                fail_count += 1
                logger.error("✗ %s\n    %s", filename, exc.stderr)

            except Exception as exc:            # noqa: BLE001
                manifest.mark_failed(filename, str(exc))
                fail_count += 1
                logger.error("✗ %s — unexpected error: %s", filename, exc)

        # Save manifest after every batch (crash-safe)
        manifest.save()
        logger.info("Batch %d complete — done: %d, failed: %d",
                    batch_num, done_count, fail_count)

    print_report(manifest, logger)
    logger.info("Phase 2 complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 — Extract audio from Al-Shaarawi Tafsir videos"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100,
        help="Number of files to process per batch (default: 100)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be processed without running FFmpeg"
    )
    parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Process only the first N pending files"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_extraction(
    batch_size=args.batch_size,
    dry_run=args.dry_run,
    limit=args.limit
    )