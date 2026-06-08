"""
scan_videos.py
--------------
Scans the raw_videos directory and extracts structured data from each filename.

Each filename follows one of these Arabic patterns:
  - تفسير سورة الفاتحة رقم 1
  - سورة يونس حلقة رقم 38
  - مقدمة رقم 3

Returns a list of dicts with: filename, surah, episode, path.
"""

import re
import os
from pathlib import Path


# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}

# Regex patterns — ordered from most specific to least specific
# Each pattern captures: (surah_name, episode_number)
FILENAME_PATTERNS = [
    # Pattern 1: تفسير سورة الفاتحة رقم 1
    r"تفسير\s+سورة\s+([\u0600-\u06FF\s]+?)\s+رقم\s+(\d+)",

    # Pattern 2: سورة يونس حلقة رقم 38
    r"سورة\s+([\u0600-\u06FF\s]+?)\s+حلقة\s+رقم\s+(\d+)",

    # Pattern 3: سورة الفاتحة رقم 1  (no حلقة)
    r"سورة\s+([\u0600-\u06FF\s]+?)\s+رقم\s+(\d+)",
]

# Special pattern for مقدمة (introduction episodes — no surah name)
INTRO_PATTERN = r"مقدمة\s+رقم\s+(\d+)"


def _extract_info_from_filename(filename: str) -> dict:
    """
    Try each regex pattern against the filename stem.
    Returns a dict with keys: surah, episode, is_intro.
    Returns None if nothing matches.
    """
    stem = Path(filename).stem  # filename without extension

    # Check for intro episodes first
    intro_match = re.search(INTRO_PATTERN, stem)
    if intro_match:
        return {
            "surah": "مقدمة",
            "episode": int(intro_match.group(1)),
            "is_intro": True,
        }

    # Try each surah pattern
    for pattern in FILENAME_PATTERNS:
        match = re.search(pattern, stem)
        if match:
            surah_name = match.group(1).strip()
            episode_num = int(match.group(2))
            return {
                "surah": surah_name,
                "episode": episode_num,
                "is_intro": False,
            }

    return None  # Could not parse


def scan_videos(videos_dir: str) -> tuple[list[dict], list[str]]:
    """
    Scan the given directory for video files.

    Args:
        videos_dir: Path to the raw_videos folder.

    Returns:
        (records, unmatched)
        - records   : list of dicts for successfully parsed files
        - unmatched : list of filenames that could not be parsed
    """
    videos_path = Path(videos_dir)

    if not videos_path.exists():
        raise FileNotFoundError(f"Directory not found: {videos_dir}")

    records = []
    unmatched = []

    for file in sorted(videos_path.iterdir()):
        # Skip non-video files
        if file.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        info = _extract_info_from_filename(file.name)

        if info is None:
            unmatched.append(file.name)
            continue

        records.append({
            "filename": file.name,
            "surah": info["surah"],
            "episode": info["episode"],
            "is_intro": info["is_intro"],
            "path": str(file.resolve()),
        })

    return records, unmatched


# ── Quick test when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    videos_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw_videos"

    print(f"Scanning: {videos_dir}\n")
    records, unmatched = scan_videos(videos_dir)

    print(f"✅ Parsed    : {len(records)} videos")
    print(f"❌ Unmatched : {len(unmatched)} videos")

    if unmatched:
        print("\n── Unmatched filenames ──")
        for name in unmatched:
            print(f"  {name}")

    if records:
        print("\n── Sample (first 5) ──")
        for r in records[:5]:
            print(f"  {r['surah']} | ep {r['episode']} | {r['filename']}")