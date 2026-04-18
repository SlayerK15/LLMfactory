# Collection System — Build Plan

## Vision

User enters a topic → agents expand it into a curated query tree → scrapers collect
documents at scale → cleaning pipeline produces a high-quality corpus → dataset builder
formats it for fine-tuning → a domain-specialist LLM is produced that beats larger
generalist models through surgical data quality.

**Core philosophy:** data quality governs LLM output more than parameter count.

---

## Goals & Constraints

| Constraint | Value |
|---|---|
| End-to-end pipeline target | ≤ 30 min (small models), ~45-60 min (14B) |
| Docs collected per run | 2,000 (default, user-selectable) |
| Paid APIs allowed | None — zero |
| Fine-tune output | Merged full model, GGUF int4 |
| Max model size | 14B params |
| Users (v1) | Single user |
| Deployment (now) | Local Linux SSH box (i5-10500H, 16GB RAM, 1650Ti, 1.5TB) |
| GPU for training | Cloud rental (Brev / RunPod A100, ~200-500 INR/run) |
| Artifact storage | AWS S3 (300 INR/mo budget) — deferred to Phase 2 |

---

## Phase Map

```
Phase 1 — Data Pipeline (CURRENT)
  ├── Project 1: collection-system   (query agent + search + scrape + store)
  ├── Project 2: cleaning-system     (dedup + filters + quality report)
  └── Project 3: dataset-forge       (chunking + Q/A synth + train/eval split)

Phase 2 — Training (DEFERRED)
  └── Project 4: trainer-service     (Unsloth + QLoRA + S3 upload)

Phase 3 — Product (DEFERRED)
  └── Project 5: orchestrator + API + frontend
```

Build and validate each phase completely before moving to the next.
Fine-tuning is not touched until Phase 1 data pipeline is flawless and measurable.

---

## Phase 1 — Project 1: `collection-system`

### What it does
Topic string → ranked query list → URLs discovered → pages scraped →
docs stored in PostgreSQL + filesystem.

### Tech stack (all pinned)

| Layer | Choice | Version |
|---|---|---|
| Language | Python | 3.13.13 |
| Package manager | uv | 0.11.7 |
| Agent framework | LangGraph | 1.1.6 |
| Planner LLM | Groq (llama-3.3-70b-versatile) | SDK 1.1.2 |
| Planner fallback | Ollama (qwen2.5:7b) | local |
| Search — primary | Common Crawl CDX | cdx-toolkit 0.9.38 |
| Search — secondary | SearXNG (self-hosted) | 2026.2.27 |
| Search — fallback | DuckDuckGo html/lite | disabled by default |
| Scraping | Crawl4AI | 0.8.6 |
| HTTP client | httpx | 0.28.1 |
| Validation | Pydantic | 2.13.2 |
| Config | pydantic-settings | 2.13.1 |
| Database | PostgreSQL | 18.3 |
| ORM | SQLAlchemy async | 2.0.49 |
| DB driver | asyncpg | 0.31.0 |
| Migrations | Alembic | 1.18.4 |
| Retry | tenacity | 9.1.4 |
| Circuit breaker | pybreaker | 1.4.1 |
| Metrics | prometheus-client | 0.25.0 |
| Logs | structlog (JSON → Loki) | 25.5.0 |
| CLI | typer | 0.24.1 |
| Tests | pytest + pytest-asyncio | 9.0.3 / 1.3.0 |
| Linter | ruff | 0.15.11 |
| Type checker | mypy | 1.20.1 |

### Docker Compose services

| Service | Image | Port | Storage |
|---|---|---|---|
| postgres | postgres:18.3 | 5432 | /mnt/nvme/postgres |
| searxng | searxng/searxng:2026.2.27 | 8888 | — |
| prometheus | prom/prometheus:v3.11.0 | 9090 | /mnt/nvme/prometheus |
| loki | grafana/loki:3.7.1 | 3100 | /mnt/nvme/loki |
| promtail | grafana/promtail:3.7.1 | — | ships app logs → Loki |
| grafana | grafana/grafana:13.1.0 | 3000 | /mnt/sata/grafana |
| collection-system | (our build) | 8000, 8001 | /mnt/nvme/data |

Everything self-hosted. No external SaaS. Single `docker compose up` starts the full stack.

### Architecture — hexagonal (ports + adapters)

```
collection_system/
├── core/            # zero I/O — pure business logic
│   ├── models.py    # all Pydantic domain models
│   ├── ports.py     # Protocol interfaces (LLMPort, SearchPort, ScraperPort, StoragePort)
│   ├── pipeline.py  # run_collection() orchestration
│   └── errors.py    # typed error hierarchy
├── agents/          # LangGraph query agent
│   ├── graph.py     # StateGraph: expand → flatten → score(fan-out) → filter
│   ├── nodes.py
│   ├── state.py
│   └── prompts.py
├── adapters/        # swappable I/O implementations
│   ├── search/
│   │   ├── cc_cdx.py
│   │   ├── searxng.py
│   │   ├── ddg_lite.py
│   │   └── composite.py   # fans out to enabled backends + merges
│   ├── scraper/
│   │   ├── crawl4ai_adapter.py
│   │   └── httpx_fallback.py
│   ├── storage/
│   │   ├── postgres.py
│   │   └── filesystem.py
│   └── llm/
│       ├── groq_adapter.py
│       └── ollama_adapter.py
├── infra/           # cross-cutting
│   ├── config.py
│   ├── logging.py
│   ├── metrics.py
│   ├── retry.py
│   ├── rate_limiter.py
│   ├── circuit_breaker.py
│   ├── checkpoint.py
│   └── db.py
├── cli.py
└── api.py           # public async-iterator API for future orchestrator
```

**Rule:** `core/` never imports from `adapters/` or `infra/`. Adapters are injected.
Adding a new search backend = one file in `adapters/search/` + register in config. Zero core changes.

### LangGraph query agent

```
START
  → ExpandNode        calls LLM to expand topic into sub-topics, depth-bounded
    (loops until depth == max_depth OR query count >= max_queries)
  → FlattenNode       tree → flat leaf list
  → ScoreNode         fan-out via Send(), batches of 20, scores relevance 0.0–1.0
  → FilterNode        threshold + top-N cap
  → END → list[Query]
```

LangGraph checkpointer: `AsyncPostgresSaver` — agent state survives crashes.

### Domain models (key schemas)

```
RunConfig       topic, doc_count, max_depth, max_queries, relevance_threshold,
                search_backends, scraper_concurrency, per_url_timeout_s, run_id

Query           id, run_id, text, parent_id, depth, relevance_score, source

DiscoveredURL   id, run_id, query_id, url, domain, source_backend, status

ScrapedDoc      id, run_id, url_id, url, title, markdown, content_hash,
                token_count, extraction_confidence, scraped_at, path

Failure         id, run_id, stage, target, error_type, error_msg, retries

RunManifest     run_id, config, status, stages: dict[Stage, StageStats]
```

### Port contracts (what every adapter must implement)

```
LLMPort:       expand_topic(), score_relevance(), health_check()
SearchPort:    discover_urls(), health_check(), rate_limit
ScraperPort:   scrape(), scrape_batch() → AsyncIterator, health_check()
StoragePort:   save_run/query/url/doc/failure(), load_run(), list_runs(),
               load_docs(), url_seen(), content_hash_seen()
```

### PostgreSQL schema (6 tables)

```sql
runs          (id, topic, config JSONB, status, started_at, completed_at)
queries       (id, run_id FK, text, parent_id, depth, relevance_score)
urls          (id, run_id FK, query_id FK, url, url_hash UNIQUE per run, domain, status)
docs          (id, run_id FK, url_id FK, content_hash UNIQUE per run, path, token_count)
failures      (id, run_id FK, stage, target, error_type, retries)
stage_stats   (id, run_id FK, stage, started_at, completed_at, input/output/failure counts)
```

`url_hash` unique index = free dedup at collection time (SHA256 of normalised URL).
`content_hash` unique index = free exact-content dedup at storage time.

### Pipeline execution flow

```
1. INIT             save run to DB, emit RunStarted
2. QUERY_GEN        LangGraph agent → 500-800 ranked queries → save to DB
                    checkpoint saved
3. URL_DISCOVERY    composite search fans out across backends (rate-limited per backend)
                    URL-hash dedup → save unique URLs
                    checkpoint saved
4. SCRAPE           crawl4ai.scrape_batch(concurrency=40) async streaming
                    per-doc: content-hash dedup → write .md to filesystem → save to DB
                    failures written to DB, run continues
                    checkpoint every 100 docs
5. FINALIZE         write manifest.json + metrics.json, update run status
```

**Resumability:** `collect resume <run_id>` loads checkpoint, skips done stages.
**Partial failures:** never abort a run on single-URL failure — write to failures table, continue.
**Timeouts:** per-URL 30s watchdog, per-stage budgets, global 1800s hard kill.

### Reliability mechanisms

| Concern | Solution |
|---|---|
| Transient network errors | tenacity: exponential backoff + jitter, 3 retries |
| Upstream outages | pybreaker: trips after 5 failures, 60s cooldown per backend |
| Rate limiting | Per-backend async token bucket (config: `"2/s"` for SearXNG) |
| Crashed runs | PostgreSQL checkpoint → `collect resume` |
| Duplicate URLs | SHA256(normalised_url) UNIQUE index |
| Duplicate content | SHA256(content) UNIQUE index |
| Hanging requests | Per-URL asyncio timeout watchdog |
| Graceful shutdown | SIGINT handler flushes checkpoint before exit |

### Observability

**Prometheus metrics (`:8001/metrics`):**
```
collection_queries_generated_total{run_id}
collection_urls_discovered_total{run_id, backend}
collection_docs_saved_total{run_id}
collection_failures_total{run_id, stage, error_type}
collection_scrape_duration_seconds{run_id}        (histogram)
collection_stage_duration_seconds{run_id, stage}  (histogram)
search_backend_up{backend}
circuit_breaker_state{backend}
rate_limit_wait_seconds{backend}
```

**Loki logs:** structured JSON per event, always carrying `run_id + stage + level`.

### CLI commands

```bash
collect run "DevOps"              # start a run
collect run "DevOps" --doc-count 500 --concurrency 20

collect resume <run_id>           # resume from last checkpoint

collect status <run_id>           # stage-by-stage progress table
collect list                      # all runs: status + doc counts + timing
collect failures <run_id>         # error log with types
collect inspect <run_id>          # full manifest

collect health                    # ping all adapters
collect db migrate                # alembic upgrade head

collect serve                     # FastAPI server (for Phase 3 orchestrator)
```

### Public API (frozen contract — orchestrator will consume this)

```python
from collection_system import run_collection, RunConfig

# blocking
handle = await run_collection(RunConfig(topic="DevOps", doc_count=2000))

# streaming (what orchestrator + SSE frontend will use)
async for event in run_collection_streaming(config):
    match event:
        case QueriesGenerated(count=n): ...
        case DocScraped(url=u, token_count=t): ...
        case StageCompleted(stage=s, stats=st): ...
        case RunCompleted(stats=st): ...
        case RunFailed(error=e): ...
```

### Test plan

| Suite | What | Speed |
|---|---|---|
| `unit/test_pipeline.py` | Full pipeline with in-memory fakes for all ports | fast |
| `unit/test_query_agent.py` | LangGraph graph with FakeLLM, expansion depth, score filter | fast |
| `unit/test_rate_limiter.py` | Token bucket math + concurrent access | fast |
| `unit/test_models.py` | Pydantic validation edge cases | fast |
| `integration/test_cc_cdx.py` | Real CDX query, marked `slow` | ~15s |
| `integration/test_searxng.py` | Real SearXNG, marked `slow` | ~10s |
| `integration/test_crawl4ai.py` | Scrape 3 stable URLs, marked `slow` | ~30s |
| `integration/test_postgres.py` | Full CRUD, marked `slow` | ~5s |
| `golden/test_devops_queries.py` | "DevOps" → ≥200 queries, expected distribution | ~40s |

CI: `pytest -m "not slow"`. Integration: `pytest -m slow` on demand.

### Data layout on disk

```
data/
├── runs/
│   └── {run_id}/
│       ├── manifest.json          run config + final stage stats
│       ├── queries.jsonl          all queries with lineage
│       ├── urls.jsonl             all URLs + source backend
│       ├── docs/
│       │   ├── {doc_id}.md        Crawl4AI markdown output
│       │   └── {doc_id}.meta.json url, timestamp, confidence
│       ├── failed.jsonl           failed URLs with typed errors
│       └── metrics.json           per-stage timings + counts
└── checkpoints/
    └── {run_id}.json              resume state
```

---

## Phase 1 — Project 2: `cleaning-system`

**Input:** `data/runs/{run_id}/docs/`
**Output:** cleaned corpus + `reports/{run_id}.json`

### Pipeline stages (in order)

```
1. Exact dedup        SHA256 content hash (already done at collection — pass-through)
2. Near-dup           MinHash LSH (datasketch 1.9.0), Jaccard threshold 0.8
3. Language filter    fastText langid — keep target language only
4. Quality heuristics Gopher rules: min/max length, symbol ratio,
                      repetitive-line ratio, stop-word density
5. Perplexity filter  Small local LM (TinyLlama or Qwen-0.5B on CPU),
                      drop statistical outliers (both tails)
6. Relevance check    BGE-M3 embeddings, cosine sim to topic centroid,
                      drop docs below threshold
7. Artifact cleanup   Crawl4AI catches most nav/footer — Trafilatura re-extracts
                      any doc flagged as low-confidence
```

### Quality report (`reports/{run_id}.json`)

```json
{
  "run_id": "...",
  "topic": "DevOps",
  "input_docs": 1612,
  "after_near_dedup": 1453,
  "after_lang_filter": 1421,
  "after_gopher": 1287,
  "after_perplexity": 1201,
  "after_relevance": 1087,
  "total_tokens": 4231000,
  "median_doc_tokens": 3890,
  "cleaning_duration_s": 135
}
```

This report is the feedback loop. Run same topic twice, watch the numbers move,
iterate on filter thresholds without touching code.

---

## Phase 1 — Project 3: `dataset-forge`

**Input:** cleaned corpus from `cleaning-system`
**Output:** `datasets/{run_id}.jsonl` — training-ready dataset

### Two output modes (user selects)

**Continued pretraining** (raw LM loss):
- Chunk docs to ≤2048 tokens with overlap
- Export as `{"text": "..."}` JSONL
- No LLM calls needed — pure formatting
- Fastest path to training

**Instruction tuning** (Q/A pairs):
- Generate 3 Q/A pairs per doc via Groq (batched 20 docs/call)
- Filter pairs below quality threshold
- Export as `{"instruction": "...", "input": "", "response": "..."}` JSONL
- Produces a proper domain assistant
- Adds 5-15 min depending on Groq rate limits

### Also emits
- BGE-M3 embeddings → LanceDB collection (for future RAG use, free)
- Train/eval split (default 90/10)
- Dataset card (`datasets/{run_id}.card.json`): topic, docs count, tokens,
  format, split sizes, source backends, cleaning stats

---

## Phase 2 — Project 4: `trainer-service` (DEFERRED)

Not started until Phase 1 is validated end-to-end.

Planned: Modal serverless function, Unsloth + QLoRA, merged GGUF int4 output,
S3 artifact upload. User selects base model (any HF model ≤14B), training style,
epochs, LoRA rank. Auto-selects biggest model that fits within user's time budget.

---

## Phase 3 — Orchestrator + Frontend (DEFERRED)

Minimal FastAPI orchestrator consuming the frozen `AsyncIterator[CollectionEvent]`
API from `collection-system`. Simple HTML form → POST → SSE log stream as v0 UI,
upgradeable to Next.js without API changes.

---

## 30-Minute Pipeline Budget (reference)

| Stage | Optimistic | Realistic |
|---|---|---|
| Query generation (Groq) | 0:30 | 1:00 |
| URL discovery (CC CDX + SearXNG) | 1:00 | 2:30 |
| Scraping 2k URLs (Crawl4AI, 40 conc.) | 3:00 | 8:00 |
| Cleaning (Phase 2) | 1:00 | 3:00 |
| Dataset build — pretraining | 0:30 | 2:00 |
| Dataset build — instruct tuning | 3:00 | 10:00 |
| Fine-tune 3B on cloud A100 | 3:00 | 6:00 |
| Fine-tune 7B on cloud A100 | 8:00 | 15:00 |
| Fine-tune 14B on cloud A100 | 18:00 | 28:00 |
| Quantize + upload | 3:00 | 6:00 |

**Model-size recommendation:**
- 30-min guarantee → cap at 7B
- 45-60 min acceptable → 14B works
- User selects; system auto-warns if budget is tight

---

## Environment variables (`.env.example`)

```bash
# LLM
GROQ_API_KEY=gsk_...
OLLAMA_BASE_URL=http://localhost:11434

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@postgres:5432/collection
POSTGRES_USER=collection
POSTGRES_PASSWORD=changeme

# Config
CONFIG_FILE=configs/prod.toml
DATA_DIR=/app/data
```

---

## Key external resources

| Resource | URL / identifier |
|---|---|
| Common Crawl current index | CC-MAIN-2026-12 (via index.commoncrawl.org) |
| BGE-M3 embeddings | BAAI/bge-m3 on HuggingFace |
| Groq model | llama-3.3-70b-versatile |
| SearXNG docs | docs.searxng.org |
| LangGraph docs | python.langchain.com/docs/langgraph |

---

*Plan version: 2026-04-17. Phases 2 and 3 are intentionally sparse — detail added when Phase 1 is complete.*
