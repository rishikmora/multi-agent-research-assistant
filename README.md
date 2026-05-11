# Multi-Agent Research Assistant

Production-grade research pipeline powered by 5 specialist AI agents, LangGraph orchestration, FastAPI, Next.js, and a full observability stack.

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         Orchestrator Agent          │  Query analysis, complexity scoring
└─────────────────┬───────────────────┘
                  │
    ┌─────────────▼───────────────┐
    │         Planner Agent       │  Subtask decomposition, scope enforcement
    └─────────────┬───────────────┘
                  │ (parallel)
    ┌─────────────▼───────────────┐
    │  Researcher A ∥ Researcher B│  Web + academic search, semaphore-limited
    └─────────────┬───────────────┘
                  │
    ┌─────────────▼───────────────┐
    │         Critic Agent        │  Gap analysis, confidence scoring
    └──────┬──────────────┬───────┘
           │ refine        │ done
    (loop≤2)              │
           └──────────────┼──→ Synthesizer → Report
                          ▼
                  Structured Report
                  (sections + citations + confidence scores)
```

## What makes this different

| Feature | Ours | CrewAI | AutoGen | LangGraph |
|---|---|---|---|---|
| Structural verification (separate Critic) | ✓ | ✗ | ~ | ~ |
| Confidence scoring per section | ✓ | ✗ | ✗ | ✗ |
| Bounded refinement with hard cap | ✓ | ✗ | ✗ | ✓ |
| Task scope enforcement (no overlap) | ✓ | ~ | ✗ | ~ |
| SSE streaming agent events | ✓ | ✗ | ✗ | ~ |
| Full Prometheus + Grafana observability | ✓ | ~ | ✗ | ~ |
| pgvector semantic memory | ✓ | ✗ | ✗ | ~ |

## Tech stack

**Backend**
- FastAPI 0.115 + Uvicorn + Gunicorn (4 workers)
- LangGraph 0.2 — stateful graph with PostgreSQL checkpointing
- Anthropic Claude (Opus for orchestration, Sonnet for research/critique)
- PostgreSQL 16 + pgvector — persistent state + semantic search
- Redis 7 — session queues, SSE fan-out, token budget tracking
- Prometheus + structlog — full observability

**Frontend**
- Next.js 15 (App Router) + TypeScript
- Zustand — SSE-connected pipeline state
- Framer Motion — agent step animations
- Tailwind CSS

**Infrastructure**
- Docker Compose (dev) / Kubernetes (prod)
- Nginx — SSE-aware reverse proxy (buffering disabled on stream endpoints)
- Grafana dashboards — LLM latency, token burn, pipeline throughput

## Quickstart

```bash
# 1. Clone and configure
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY at minimum

# 2. Start all services
docker compose up -d

# 3. Verify
curl http://localhost:8000/health

# 4. Open the UI
open http://localhost:3000/research
```

## API

### Start research
```bash
POST /api/v1/research
Content-Type: application/json

{
  "query": "Your research question",
  "depth": "standard",        # quick | standard | deep
  "max_sources": 20,
  "include_arxiv": true
}

# Returns: { "session_id": "uuid", "status": "queued" }
```

### Stream live events
```bash
GET /api/v1/research/{session_id}/stream
# Server-Sent Events stream with typed events:
# pipeline_start | agent_start | agent_progress | agent_complete
# sources_found | refinement_loop | report_section
# pipeline_complete | pipeline_error
```

### Get completed report (polling fallback)
```bash
GET /api/v1/research/{session_id}
```

## Development

```bash
# Backend
cd backend
poetry install
cp ../.env.example .env
uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev

# Tests
cd backend
pytest tests/ -v --cov=app --cov-report=term-missing

# Type checking
mypy app/
```

## Production deployment

### Environment checklist
- [ ] `SECRET_KEY` is 32+ chars, randomly generated
- [ ] `ANTHROPIC_API_KEY` is set
- [ ] `ENVIRONMENT=production` (disables /docs, /openapi.json)
- [ ] `POSTGRES_PASSWORD` is strong and unique
- [ ] `CORS_ORIGINS` is restricted to your actual domain
- [ ] `BRAVE_API_KEY` or `SERPAPI_KEY` for real search results
- [ ] `LANGFUSE_*` keys for LLM tracing
- [ ] `SENTRY_DSN` for error tracking

### Kubernetes
```bash
kubectl apply -f infra/k8s/
```

### Scaling notes
- Horizontal scale the backend with `WORKERS` env var or K8s replicas
- Redis must be shared across all backend instances for SSE consistency
- LangGraph checkpoints go to PostgreSQL — no in-memory state
- With Redis pub/sub, SSE works across multiple backend pods

## Key design decisions

**Why SSE over WebSockets?**
Agent events are strictly server→client. SSE is simpler, HTTP/1.1 compatible, auto-reconnects, and works through nginx without sticky sessions.

**Why bounded refinement (max 2)?**
Production research from Anthropic's own multi-agent team shows sparse topics cause infinite refinement loops. Hard-capping at 2 iterations, then noting gaps in the report, is the right tradeoff.

**Why separate Critic?**
Shared context between researcher and verifier creates correlated errors — the model that wrote the findings will tend to verify them. Structural separation is the only reliable fix.

**Why LangGraph over CrewAI?**
LangGraph gives us deterministic graph edges, PostgreSQL checkpointing (resumable on failure), and `conditional_edges` for the refinement routing. CrewAI's role-based model is easier to start but harder to make reliable.
