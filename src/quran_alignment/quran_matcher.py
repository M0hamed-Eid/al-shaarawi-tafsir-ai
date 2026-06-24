"""
quran_matcher.py  —  backward-compatible convenience shim.

The matching logic now lives in dedicated modules. This re-exports the public
surface so older imports keep working:

    from quran_alignment.quran_matcher import Enricher, QuranIndex
"""
from .enrich import Enricher          # noqa: F401
from .quran_index import QuranIndex   # noqa: F401
from .config import load_config       # noqa: F401

__all__ = ["Enricher", "QuranIndex", "load_config"]
