"""
metadata_builder.py
-------------------
Combines scan_videos + surah_order.json to produce:
  - data/metadata/metadata.json   (full index, sorted by Quran order)
  - data/metadata/statistics.json (episode count per Surah)

Run:
    python src/ingestion/metadata_builder.py
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Import scanner from same package
import sys
sys.path.insert(0, str(Path(__file__).parent))
from scan_videos import scan_videos


# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR       = Path(__file__).resolve().parents[2]   # project root
VIDEOS_DIR     = ROOT_DIR / "data" / "raw_videos"
METADATA_DIR   = ROOT_DIR / "data" / "metadata"
SURAH_ORDER_FILE = ROOT_DIR / "data" / "metadata" / "surah_order.json"

OUTPUT_METADATA   = METADATA_DIR / "metadata.json"
OUTPUT_STATISTICS = METADATA_DIR / "statistics.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_surah_order(path: Path) -> dict:
    """
    Load surah_order.json and return a dict:
        { "الفاتحة": 1, "البقرة": 2, ... }
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return {entry["name"]: entry["order"] for entry in data["surahs"]}


def _normalize_surah_name(name: str) -> str:
    """
    Light normalization to handle minor spelling differences in filenames.
    Example: 'ال إسراء' → 'الإسراء'
    """
    return name.replace("ال ", "ال").strip()


def build_metadata(videos_dir: Path, surah_order: dict) -> tuple[list[dict], list[str]]:
    """
    Scan videos, enrich each record with surah_order, sort by (surah_order, episode).

    Returns:
        (metadata_records, unmatched_filenames)
    """
    records, unmatched = scan_videos(str(videos_dir))

    enriched = []
    no_order_found = []

    for record in records:
        surah_name = record["surah"]
        normalized = _normalize_surah_name(surah_name)

        order = surah_order.get(surah_name) or surah_order.get(normalized)

        if order is None and not record["is_intro"]:
            no_order_found.append(record["filename"])
            order = 9999  # push unknown surahs to the end

        enriched.append({
            "filename":    record["filename"],
            "surah":       surah_name,
            "surah_order": order if order is not None else 0,  # 0 = مقدمة
            "episode":     record["episode"],
            "is_intro":    record["is_intro"],
            "path":        record["path"],
        })

    # Sort by surah_order, then by episode number within each surah
    enriched.sort(key=lambda r: (r["surah_order"], r["episode"]))

    all_unmatched = unmatched + no_order_found
    return enriched, all_unmatched


def build_statistics(metadata: list[dict]) -> dict:
    """
    Count episodes per Surah.
    Returns an OrderedDict sorted by surah_order.
    """
    counts: dict[str, dict] = {}

    for record in metadata:
        surah = record["surah"]
        if surah not in counts:
            counts[surah] = {
                "surah_order": record["surah_order"],
                "episode_count": 0,
            }
        counts[surah]["episode_count"] += 1

    # Sort by surah_order
    sorted_stats = dict(
        sorted(counts.items(), key=lambda item: item[1]["surah_order"])
    )
    return sorted_stats


def print_statistics(stats: dict) -> None:
    """Pretty-print statistics to the terminal."""
    print("\n" + "═" * 45)
    print("  📊  Archive Statistics")
    print("═" * 45)

    total = 0
    for surah, info in stats.items():
        order = info["surah_order"]
        count = info["episode_count"]
        total += count
        label = f"{order:>3}. {surah}"
        print(f"  {label:<25} {count:>4} episodes")

    print("─" * 45)
    print(f"  {'Total':<25} {total:>4} videos")
    print("═" * 45 + "\n")


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("🔍 Loading Surah order map...")
    surah_order = load_surah_order(SURAH_ORDER_FILE)

    print(f"📂 Scanning videos in: {VIDEOS_DIR}")
    metadata, unmatched = build_metadata(VIDEOS_DIR, surah_order)

    print(f"\n✅ Indexed   : {len(metadata)} videos")
    print(f"❌ Unmatched : {len(unmatched)} videos")

    if unmatched:
        print("\n── Could not process ──")
        for name in unmatched:
            print(f"  ⚠️  {name}")

    # Build and print statistics
    stats = build_statistics(metadata)
    print_statistics(stats)

    # Wrap metadata in a top-level object with a timestamp
    output = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "total_videos": len(metadata),
        "videos": metadata,
    }

    # Save outputs
    save_json(output, OUTPUT_METADATA)
    save_json(stats,  OUTPUT_STATISTICS)

    print("\n🎉 Phase 1 complete. Metadata ready for Phase 2.")


if __name__ == "__main__":
    main()