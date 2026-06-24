"""
config.py
---------
Typed access to config/enrichment.yaml. All thresholds are read through this
object so nothing is hard-coded in the algorithm modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project root = two levels up from this file (src/quran_alignment/config.py)
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "enrichment.yaml"


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


@dataclass
class Config:
    raw: dict[str, Any] = field(default_factory=dict)

    # ---- path helpers --------------------------------------------------
    @property
    def quran_xml(self) -> Path:   return _abs(self.raw["paths"]["quran_xml"])
    @property
    def quran_db(self) -> Path:    return _abs(self.raw["paths"]["quran_db"])
    @property
    def quran_db_jsonl(self) -> Path: return _abs(self.raw["paths"]["quran_jsonl"])
    @property
    def transcripts_dir(self) -> Path: return _abs(self.raw["paths"]["transcripts_dir"])
    @property
    def output_dir(self) -> Path:  return _abs(self.raw["paths"]["output_dir"])
    @property
    def surah_order(self) -> Path: return _abs(self.raw["paths"]["surah_order"])
    @property
    def log_dir(self) -> Path:     return _abs(self.raw["paths"]["log_dir"])

    # ---- sub-sections (plain dicts, accessed by the algorithms) --------
    @property
    def norm(self) -> dict[str, Any]:   return self.raw["normalization"]
    @property
    def retrieval(self) -> dict[str, Any]: return self.raw["retrieval"]
    @property
    def verify(self) -> dict[str, Any]: return self.raw["verification"]
    @property
    def multiverse(self) -> dict[str, Any]: return self.raw["multiverse"]
    @property
    def output(self) -> dict[str, Any]: return self.raw["output"]


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(raw=raw)
