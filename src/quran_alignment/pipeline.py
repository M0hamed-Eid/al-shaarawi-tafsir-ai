"""
pipeline.py  —  batch enrichment over all 1214 transcripts
==========================================================

* Builds the QuranIndex ONCE per worker (corpus + bigram + BM25, ~0.3 s).
* Streams every data/transcripts/**/*.json through the Enricher.
* Writes one enriched JSON per episode mirroring the input tree, plus a
  corpus-level matches.jsonl (every accepted/review match, one per line) and a
  run_stats.json summary.

SCALING
-------
~0.2 s/episode single-core  ->  ~4 min for 1214 episodes on one core.
`--jobs N` fans out with a process pool (each worker holds its own index), so on
a typical laptop the whole corpus finishes in ~1 minute. Memory: the corpus +
indexes are ~60 MB per worker. Everything is CPU-only; no GPU, no network.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

from .config import Config, load_config
from .enrich import Enricher

# one Enricher per process (lazy global so ProcessPool workers reuse it)
_WORKER: Optional[Enricher] = None
_CFG_PATH: Optional[str] = None


def _worker_init(cfg_path: Optional[str]):
    global _WORKER, _CFG_PATH
    _CFG_PATH = cfg_path
    _WORKER = Enricher(load_config(cfg_path))


def _worker_run(in_path: str) -> dict:
    assert _WORKER is not None
    cfg = _WORKER.cfg
    in_p = Path(in_path)
    try:
        res = _WORKER.enrich_file(in_p)
    except Exception as e:  # never let one bad file kill the batch
        return {"path": in_path, "ok": False, "error": repr(e)}
    rel = in_p.relative_to(cfg.transcripts_dir)
    out_p = cfg.output_dir / rel
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(res.model_dump_json(indent=2), encoding="utf-8")
    return {
        "path": str(rel), "ok": True,
        "surah": res.surah, "episode": res.episode,
        "total": res.total_matches, "accepted": res.accepted,
        "review": res.review, "cross_surah": res.cross_surah,
        "matches": [m.model_dump() for m in res.matches],
    }


def discover(transcripts_dir: Path) -> list[Path]:
    return sorted(transcripts_dir.rglob("*.json"))


def run(cfg: Config, cfg_path: Optional[str] = None, jobs: int = 1,
        limit: Optional[int] = None) -> dict:
    files = discover(cfg.transcripts_dir)
    if limit:
        files = files[:limit]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    matches_path = cfg.output_dir / "matches.jsonl"
    stats = {"episodes": 0, "ok": 0, "failed": 0, "total_matches": 0,
             "accepted": 0, "review": 0, "cross_surah": 0, "errors": []}
    t0 = time.time()

    mf = open(matches_path, "w", encoding="utf-8")

    def handle(r: dict):
        stats["episodes"] += 1
        if not r["ok"]:
            stats["failed"] += 1
            stats["errors"].append(r)
            return
        stats["ok"] += 1
        stats["total_matches"] += r["total"]
        stats["accepted"] += r["accepted"]
        stats["review"] += r["review"]
        stats["cross_surah"] += r["cross_surah"]
        for m in r["matches"]:
            mf.write(json.dumps({"episode_path": r["path"], "primary_surah": r["surah"],
                                 **{k: m[k] for k in (
                                     "surah", "ayah", "ayah_end", "verse_id",
                                     "confidence", "status", "start", "end",
                                     "is_cross_surah", "transcript_text",
                                     "canonical_quoted")}},
                                ensure_ascii=False) + "\n")

    if jobs and jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs,
                                 initializer=_worker_init, initargs=(cfg_path,)) as ex:
            for r in ex.map(_worker_run, [str(f) for f in files], chunksize=4):
                handle(r)
    else:
        _worker_init(cfg_path)
        for f in files:
            handle(_worker_run(str(f)))

    mf.close()
    stats["seconds"] = round(time.time() - t0, 2)
    (cfg.output_dir / "run_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats
