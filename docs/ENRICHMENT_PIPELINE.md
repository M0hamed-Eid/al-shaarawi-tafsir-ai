# Quran Reference Enrichment Pipeline (Phase 4)

Detect Quran quotations inside Al-Shaarawi Whisper transcripts and replace them
with the canonical **Tanzil Uthmani** text (with tashkeel), tagged with surah,
ayah number(s), confidence and **preserved start/end timestamps**.

```json
{ "surah": "البقرة", "ayah": 173,
  "canonical_text": "إِنَّ ٱللَّهَ غَفُورٌ رَّحِيمٌ",
  "confidence": 0.87, "start": 123.0, "end": 127.0,
  "status": "accepted", "is_cross_surah": false }
```

Input  : `قال الله تعالى ان الله غفور رحيم ثم يكمل الشرح`
Output : `قال الله تعالى ﴿ إِنَّ ٱللَّهَ غَفُورٌ رَّحِيمٌ ﴾ [البقرة:173] ثم يكمل الشرح`

---

## 1. Architecture (ASCII)

```
                 ┌──────────────────────────────────────────────────────────┐
 BUILD (once)    │  Tanzil Uthmani XML  (bundled in pyquran, copied to data) │
                 │        │  parse 6236 ayat                                  │
                 │        ▼                                                   │
                 │  data/quran/quran.jsonl   id│surah│ayah│uthmani│norm│align │
                 │        │                                                   │
                 │        ▼ (load into RAM, one-time ~0.3s)                   │
                 │  QuranIndex:  verses[]  +  BIGRAM inverted index  +  BM25  │
                 └──────────────────────────────────────────────────────────┘
                                          │  reused across all 1214 episodes
 ┌────────────────────────────────────────┼──────────────────────────────────┐
 │ PER EPISODE                             ▼                                   │
 │  transcript.json ─► flatten_words ─► [Word(raw,norm,seg,pos,start,end), …]  │
 │                       (STAGE A: normalize each token)                       │
 │                                         │                                   │
 │   STAGE B  candidate gen  ◄─────────────┘                                   │
 │   for each transcript bigram → bigram_index → (verse, position) seeds       │
 │                                         │                                   │
 │   STAGE C  verify + localise            ▼                                   │
 │   seed-and-extend per seed → exact word-span + rapidfuzz.ratio              │
 │                                         │   RawMatch[]                      │
 │   STAGE E  decide (GLOBAL, cross-surah) ▼                                   │
 │   confidence + primary-surah boost + ambiguity guard + disjoint greedy      │
 │                                         │   VerifiedMatch[]                 │
 │   STAGE D  multiverse                   ▼                                   │
 │   chain consecutive ayat + FORCE-ALIGN missed neighbours                    │
 │                                         │                                   │
 │   REPLACE  inline ﴿quoted﴾ [surah:ayah] ▼  (timestamps untouched)           │
 │                                         │                                   │
 │   EnrichedTranscript  ─►  data/enriched/<surah>/<ep>.json                   │
 └────────────────────────────────────────────────────────────────────────────┘
        Batch driver (pipeline.py): ProcessPool over episodes  +  matches.jsonl
```

---

## 2. Folder structure

```
al-shaarawi-tafsir-ai/
├── config/
│   └── enrichment.yaml             # every threshold / path / toggle
├── data/
│   ├── quran/
│   │   ├── quran-uthmani.xml        # Tanzil source (copied from pyquran)
│   │   ├── quran.jsonl              # built corpus (6236 verses)  ← hot path
│   │   └── quran.sqlite3            # OPTIONAL FTS5 store (ad-hoc SQL)
│   ├── transcripts/<surah>/ep_NNN.json     # Phase-3 Whisper output (INPUT)
│   └── enriched/<surah>/ep_NNN.json        # pipeline OUTPUT
│       ├── matches.jsonl            # every match, one per line (review/BI)
│       └── run_stats.json           # batch summary
├── src/quran_alignment/
│   ├── __init__.py                  # lazy package surface
│   ├── config.py                    # typed YAML access
│   ├── normalize.py                 # STAGE A
│   ├── quran_index.py               # corpus build/load + bigram + BM25 (STAGE B backend)
│   ├── transcript_io.py             # flatten Whisper -> timed Word stream
│   ├── retrieval.py                 # STAGE B (seed generation)
│   ├── verify.py                    # STAGE C (seed-extend + scoring)
│   ├── decide.py                    # STAGE E (cross-surah, ambiguity, dedup)
│   ├── multiverse.py                # STAGE D (chaining + forced alignment)
│   ├── replace.py                   # inline replacement (timestamp-safe)
│   ├── schema.py                    # Pydantic: RawMatch/VerifiedMatch/EnrichedTranscript
│   ├── enrich.py                    # orchestrator (Enricher)
│   ├── pipeline.py                  # batch + parallelism
│   └── cli.py                       # `python -m quran_alignment.cli ...`
├── scripts/build_quran_db.py        # build the corpus
├── evaluation/
│   ├── build_gold.py                # bootstrap gold sheet from predictions
│   ├── evaluate.py                  # Precision / Recall / F1 / FP / FN
│   └── gold.csv                     # human-annotated truth (you create this)
├── tests/test_enrichment.py         # 11 unit + integration tests
└── docs/ENRICHMENT_PIPELINE.md      # this file
```

---

## 3. Required Python libraries

| library     | role                                              | why |
|-------------|---------------------------------------------------|-----|
| pyquran     | ships the Tanzil Uthmani XML                      | offline canonical text, no download |
| pyarabic    | pyquran dependency                                | — |
| rapidfuzz   | C++ fuzzy string matching (Stage C)               | ~100× python-Levenshtein, has alignment API |
| rank-bm25   | in-RAM BM25                                       | search feature + optional retrieval backend |
| pydantic v2 | typed output schemas + JSON (de)serialisation     | validation + `model_dump_json` |
| PyYAML      | config                                            | — |
| tqdm        | progress                                          | — |

CPU-only, no GPU, no paid APIs, no network at run time. Managed via `pyproject.toml` + uv: `uv sync`.

---

## 4. Data source — recommendation

**Use Tanzil Uthmani.** Rationale:

* **Tanzil is the upstream** that most other corpora are derived from
  (Quran.com / api.quran.com, alquran.cloud, fawazahmed0/quran-api all trace
  back to Tanzil text). Using the source avoids a second-hand transcription
  layer and its diff risk.
* It provides a true **Uthmani edition with full tashkeel** and a documented,
  stable numbering of all **6236** ayat — exactly what we must emit as
  `canonical_text`.
* It is **plain XML/TXT, offline, redistributable** for non-commercial Quranic
  use. It ships inside the `pyquran` PyPI package, so no live download is needed.

| source            | text quality                  | access            | verdict |
|-------------------|-------------------------------|-------------------|---------|
| **Tanzil Uthmani**| canonical, full tashkeel      | offline (pyquran) | **CHOSEN** |
| Quran.com / API   | excellent, derived from Tanzil| HTTP API (network)| great for app search, network dependency for bulk text |
| alquran.cloud     | derived from Tanzil           | HTTP API          | redundant given offline Tanzil |
| fawazahmed0 JSON  | derived; some spacing quirks  | CDN download      | fine as fallback only |

We keep **one** source of truth (Uthmani). The "simple/clean" rasm used for
matching is derived on the fly by the normalisation layer, so the two never
drift apart.

---

## 5. The matching pipeline, stage by stage

### Stage A — Normalization (`normalize.py`)
Projects BOTH the Uthmani verse and the (near-diacritic-free) ASR text into one
comparison space, while the original Uthmani is kept untouched for display.
Handles, in order: NFC → tatweel removal → diacritic removal (harakat, tanwin,
shadda, sukun, **maddah**, combining hamza seats, **dagger alef**, full Quranic
annotation block) → alef folding (أ إ آ ٱ → ا) → alef maqsura (ى → ي) →
ta-marbuta (ة → ه) → hamza folding (ؤ→و, ئ→ي, drop ء) → strip non-Arabic →
collapse spaces. All combining-mark ranges are defined by numeric codepoint so
they can never be silently corrupted in source.

### Stage B — Candidate retrieval (`retrieval.py`, backend in `quran_index.py`)
We must avoid brute-forcing 6236 verses per window. Options compared:

| approach            | build   | query/window | order-aware | dep        | verdict |
|---------------------|---------|--------------|-------------|------------|---------|
| brute-force ratio   | none    | 6236 ratios  | n/a         | none       | too slow |
| Whoosh              | disk    | ~ms          | phrase      | Whoosh     | heavy, disk I/O, slow build |
| SQLite FTS5         | disk    | ~ms          | phrase      | stdlib     | good, but file-locking flaky on network mounts; per-window SQL round-trips |
| BM25 (rank_bm25)    | ~0.3s   | ~3.3 ms      | bag-of-words| rank_bm25  | ranks well; 3.3ms × ~2300 win × 1214 ep ≈ **2.6 h** |
| **inverted BIGRAM** | ~0.2s   | ~0.5 µs      | exact phrase| **stdlib** | **~1 ms / whole episode** ✅ |

**Chosen: an in-RAM inverted index of verse word-bigrams.** A bigram shared by
transcript and verse is a strong, *position-bearing* seed (Stage C needs the
position). Bigrams occurring in `> bigram_df_cap` verses are skipped so frequent
clauses don't explode the candidate set. BM25 is still built and exposed for the
product's **search** feature and as a `backend: bm25` fallback.

### Stage C — Verification & localisation (`verify.py`)
Seeds give *approximate* positions; here we recover the **exact** transcript
word-span and score it.

Scorer comparison (the one the brief asked for):

| scorer                | order-aware | localises substring | verdict |
|-----------------------|-------------|---------------------|---------|
| ratio / Levenshtein   | yes         | no                  | used on the FINAL aligned span |
| token_set_ratio       | **no**      | no                  | rejected (Quran word order is the signal) |
| token_sort_ratio      | **no**      | no                  | rejected |
| SequenceMatcher       | yes         | partial             | correct idea, slower; rapidfuzz.ratio is the C++ equivalent |
| partial_ratio         | yes         | yes, but…           | embeds the *shorter* string fully → wrong span when the transcript holds only a fragment of a long verse |

**Chosen strategy: seed-and-extend.** Locate the seed bigram inside the verse,
then walk left/right matching tokens (small fuzzy budget for ASR slips) until
agreement runs out; score the two aligned, equal-length strings with
`rapidfuzz.ratio`. This handles a short fragment, a full verse, and (via repeated
seeding) consecutive verses — uniformly. `partial_ratio` alone mis-localised the
brief's own example (it reported all 10 transcript words as the match); seed-
extend returns exactly `ان الله غفور رحيم`.

Confidence = `(ratio/100) × length_factor(matched_tokens)` where length_factor
saturates at 5 tokens (1→0.40, 2→0.66, 3→0.85, 4→0.96, 5+→1.0). Coverage
(fraction of the verse recited) is reported but **not** penalised — reciting the
tail of a long verse is a legitimate quote.

### Stage D — Multi-verse / lost boundaries (`multiverse.py`)
Each verse is seeded independently, so 2–10 consecutive verses already surface as
separate matches. This stage (1) **chains** matches whose verses are
`(v, v+1, …)` in the same surah with adjacent spans (`gap ≤ max_gap_words`),
assigning a shared `chain_id` and the run's ayah range; and (2) **extends** chains
by **forced alignment**: a neighbour verse missed by Stage B (only common bigrams,
or ASR garble) is force-aligned against the adjoining region and inserted if it
clears `extend_threshold`. This is what makes long recitations robust to merged
segments and lost boundaries. Per-verse timestamps are retained.

### Stage E — Cross-surah references (`decide.py`)
Retrieval is **global** over all 114 surahs. The episode's primary surah is used
**only** as a tie-breaker (`primary_surah_boost`) and to resolve ambiguity — never
as a hard filter, because Al-Shaarawi quotes across surahs constantly.

False-positive prevention, layered:
1. `min_partial_score` floor at Stage B.
2. `length_factor` — short coincidences cannot score high.
3. `min_standalone_tokens` — a lone 2-word hit is never auto-accepted.
4. Disjoint greedy selection — no overlapping spans.
5. **Ambiguity guard** — many clauses recur verbatim (`إن الله غفور رحيم`,
   `فبأي آلاء ربكما تكذبان`). If a *different* verse aligns to the same span within
   `ambiguity_delta` fuzz points, confidence is reduced by `ambiguity_penalty`
   (pushing it to `review`) unless one candidate is the unique primary-surah verse.

Outcome tiers: `accepted` (≥ accept_threshold) → auto-applied & inlined;
`review` (≥ review_threshold) → kept in the structured output for a human;
`rejected` → dropped.

---

## 6. Output schema (`schema.py`, Pydantic v2)

* **RawMatch** — pre-decision candidate: verse_id, surah/ayah, transcript
  word-span, raw fuzz score, matched/verse token counts, verse-token span, timestamps.
* **VerifiedMatch** — surah (Arabic), surah_no, ayah, ayah_end, verse_id,
  `canonical_text` (full ayah), `canonical_quoted` (exact recited fragment),
  confidence, status, start/end, word span, transcript_text, fuzz_score,
  coverage, `is_cross_surah`, `chain_id`.
* **EnrichedSegment / EnrichedTranscript** — original segments + per-segment
  `enriched_text` + the match list + corpus counts.

---

## 7. CLI

```bash
# 0. install + build corpus (once)
uv sync                                       # installs deps from pyproject.toml
uv run python scripts/build_quran_db.py            # writes data/quran/quran.jsonl
python scripts/build_quran_db.py --fts      # + optional SQLite/FTS5

# 1. sanity check on the brief's example
python -m quran_alignment.cli demo

# 2. one transcript
python -m quran_alignment.cli enrich-file "data/transcripts/آل عمران/ep_001.json" \
       --out "data/enriched/آل عمران/ep_001.json"

# 3. the whole corpus (parallel)
python -m quran_alignment.cli enrich-all --jobs 8
python -m quran_alignment.cli enrich-all --limit 20      # quick subset

# 4. evaluation
python evaluation/build_gold.py --sample 25 --include-review   # -> gold_sheet.csv
#   (annotate VERDICT, add MISSED rows, save as evaluation/gold.csv)
python evaluation/evaluate.py --status accepted
python evaluation/evaluate.py --status accepted,review        # recall-leaning view
```

---

## 8. Config (`config/enrichment.yaml`)
All behaviour is data-driven; the important knobs:

* `normalization.*` — toggle each folding rule (default: all on).
* `retrieval.bigram_df_cap` (50) — anchor selectivity. ↓ = fewer false anchors.
* `verification.min_partial_score` (78), `accept_threshold` (0.82),
  `review_threshold` (0.70) — the precision/recall dial.
* `verification.ambiguity_delta` (2.0) / `ambiguity_penalty` (0.12) — recurring-clause guard.
* `verification.primary_surah_boost` (0.03) / `min_standalone_tokens` (3).
* `multiverse.max_gap_words` (4) / `extend_threshold` (0.72) / `max_chain` (12).
* `output.replace_with` (`quoted` | `full_ayah`), `keep_review_matches`.

---

## 9. Evaluation methodology

**Unit of evaluation:** `(episode, surah, ayah)` — "did we attach the right ayah
here?". Optional stricter span scoring via timestamp overlap.

**Building the gold set (the fast way):**
1. `build_gold.py` samples episodes **stratified by surah** and emits a CSV of
   predictions to *verify* (verifying is far faster than annotating from scratch).
2. A qualified annotator marks `VERDICT` = 1 (correct) / 0 (wrong) for every row.
3. **For recall**, the annotator also appends rows for any quotation the system
   **MISSED** (`status=MISSED`, `VERDICT=1`).
4. Save as `evaluation/gold.csv`.

**Metrics** (`evaluate.py`): with predictions P (status-filtered) and truth T
(VERDICT==1): `TP=|P∩T|`, `FP=|P\T|`, `FN=|T\P|`,
`Precision=TP/(TP+FP)`, `Recall=TP/(TP+FN)`, `F1=2PR/(P+R)`. It prints the
confusion counts and lists FP/FN for direct inspection. Run with
`--status accepted` (production) and `--status accepted,review` (to see the gain
from lowering the bar). Recommended gold size: ≥ 30 episodes / ≥ 400 verified
references for a stable estimate; refresh thresholds against it.

---

## 10. Testing strategy (`tests/test_enrichment.py`)
11 tests covering: Stage-A folding (maddah/dagger-alef/wasla/maqsura/ta-marbuta),
normalisation idempotence, corpus size (6236), exact quoted-fragment extraction,
seed-extend localisation, confidence monotonicity, the brief's end-to-end match
**and** replacement string, cross-surah flagging, and multi-verse chaining.
`python -m pytest tests/ -q` or `python tests/test_enrichment.py`.

---

## 11. Expected accuracy
On clear recitations with good ASR, after threshold tuning against a gold set:

* **Clear, multi-word recitations (≥4 words):** precision ~0.95+, recall ~0.90+.
  These are anchored by multiple rare bigrams and verify at ratio ≈ 100.
* **Short fragments (2–3 words) of recurring clauses:** lower — these land in
  `review` by design (ambiguity guard). Precision on `accepted` stays high;
  recall on these specific cases is the main loss.
* **Paraphrase / heavy ASR error:** out of scope for verbatim matching; recall
  drops, but these are not exact quotations.

Net expectation on `accepted` matches: **precision ≈ 0.92–0.97**, with recall
tunable via `review_threshold` (accepting review-tier trades ~5–10 precision
points for recall). Treat these as design targets to confirm on your gold set —
they depend on ASR quality per surah.

---

## 12. Scaling to 1214 transcripts
* Corpus + indexes built **once**, reused for every episode (≈ 60 MB RAM).
* Per episode ≈ **0.2 s** single-core (measured on a 3,472-word episode, 42
  matches). Whole corpus ≈ **4 min single-core**.
* `enrich-all --jobs N` fans out with a process pool (each worker holds its own
  index) → ≈ **1 min** on a typical 8-core laptop. Memory ≈ 60 MB × N.
* Fully CPU-only, no GPU, no network. Failures are isolated per file (logged in
  `run_stats.json`), so one bad transcript never aborts the batch.
* Output is incremental JSON per episode + a single `matches.jsonl` for BI /
  review tooling and for feeding the platform's search & RAG layers.
