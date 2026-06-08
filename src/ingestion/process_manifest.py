"""
process_manifest.py
-------------------
Tracks processing state across phases (extraction, transcription, etc.).
Enables resumable execution: already-completed files are skipped automatically.

Manifest schema (per phase):
{
  "phase": "extraction",
  "updated_at": "2026-06-08T...",
  "counts": { "total": 1214, "done": 0, "failed": 0, "pending": 1214 },
  "records": {
    "<filename>": {
      "status": "pending" | "done" | "failed",
      "output_path": "...",
      "error": null | "...",
      "processed_at": null | "2026-06-08T..."
    }
  }
}
"""

import json
from datetime import datetime, UTC
from pathlib import Path


class ProcessManifest:
    VALID_STATUSES = {"pending", "done", "failed"}

    def __init__(self, path: Path, phase: str):
        self.path = path
        self.phase = phase
        self._data: dict = {}
        self._load_or_init()

    # ── Public API ────────────────────────────────────────────────────────────

    def register(self, filename: str, output_path: str) -> None:
        """Add a file to the manifest if not already present."""
        if filename not in self._data["records"]:
            self._data["records"][filename] = {
                "status": "pending",
                "output_path": output_path,
                "error": None,
                "processed_at": None,
            }

    def mark_done(self, filename: str) -> None:
        self._update(filename, status="done", error=None)

    def mark_failed(self, filename: str, error: str) -> None:
        self._update(filename, status="failed", error=error)

    def is_done(self, filename: str) -> bool:
        rec = self._data["records"].get(filename)
        return rec is not None and rec["status"] == "done"

    def pending_files(self) -> list[str]:
        return [
            fn for fn, rec in self._data["records"].items()
            if rec["status"] != "done"
        ]

    def save(self) -> None:
        self._refresh_counts()
        self._data["updated_at"] = datetime.now(UTC).isoformat() + "Z"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    @property
    def counts(self) -> dict:
        self._refresh_counts()
        return dict(self._data["counts"])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_or_init(self) -> None:
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)
        else:
            self._data = {
                "phase": self.phase,
                "updated_at": datetime.now(UTC).isoformat() + "Z",
                "counts": {"total": 0, "done": 0, "failed": 0, "pending": 0},
                "records": {},
            }

    def _update(self, filename: str, *, status: str, error: str | None) -> None:
        if filename not in self._data["records"]:
            raise KeyError(f"File not registered in manifest: {filename}")
        rec = self._data["records"][filename]
        rec["status"] = status
        rec["error"] = error
        rec["processed_at"] = datetime.now(UTC).isoformat() + "Z"

    def _refresh_counts(self) -> None:
        records = self._data["records"]
        total = len(records)
        done = sum(1 for r in records.values() if r["status"] == "done")
        failed = sum(1 for r in records.values() if r["status"] == "failed")
        self._data["counts"] = {
            "total": total,
            "done": done,
            "failed": failed,
            "pending": total - done - failed,
        }