"""
Quran reference enrichment package.

Lazy attribute access (PEP 562) so importing a single submodule for testing does
not force every sibling to import.

Public surface:
    from quran_alignment import Enricher, load_config, normalize, QuranIndex
"""
_LAZY = {
    "load_config": ("config", "load_config"),
    "Config": ("config", "Config"),
    "normalize": ("normalize", "normalize"),
    "normalize_tokens": ("normalize", "normalize_tokens"),
    "QuranIndex": ("quran_index", "QuranIndex"),
    "Enricher": ("enrich", "Enricher"),
}

__all__ = list(_LAZY)


def __getattr__(name):
    import importlib
    if name in _LAZY:
        mod, attr = _LAZY[name]
        m = importlib.import_module(f"{__name__}.{mod}")
        return getattr(m, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
