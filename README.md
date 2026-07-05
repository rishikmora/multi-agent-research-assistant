# MARS Three-Tier Memory System

Production implementation of Week 1 from the intelligence upgrade roadmap.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  TIER 1 — Working Memory (Redis)                                │
│  Scope: current session only. TTL: 2 hours.                     │
│  - Agent outputs as they complete                                │
│  - Live confidence scores (atomic Lua-script updates)           │
│  - Debate transcript (bounded to 30 messages)                   │
│  - Live evidence graph state                                     │
│  - Pub/sub for cross-process SSE notification                    │
└───────────────────────┬───────────────────────────────────────────┘
                        │ finalize_session() on pipeline completion
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 2 — Episodic Memory (PostgreSQL)                          │
│  Scope: permanent record of every completed session.             │
│  - Append-only — never mutated, only superseded                  │
│  - Fast-path topic index for exact/fuzzy lookup                  │
│  - Retention: 180 days for superseded episodes (current           │
│    conclusions on any topic are kept indefinitely)                │
└───────────────────────┬───────────────────────────────────────────┘
                        │ consolidate_session() — async background task
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│  TIER 3 — Semantic Memory (PostgreSQL + pgvector)                │
│  Scope: durable, cross-session, cross-user distilled facts.       │
│  - HNSW index: m=16, ef_construction=96 (production-tuned)       │
│  - Reinforcement: corroborating episodes boost confidence         │
│    with diminishing returns (asymptotic to 0.97)                  │
│  - Decay: unreinforced facts lose confidence — half-life 90 days  │
│  - Conflicts are FLAGGED, never silently auto-resolved             │
└─────────────────────────────────────────────────────────────────┘
```

## Files

```
backend/
├── memory/
│   ├── working.py         Tier 1 — Redis working memory
│   ├── episodic.py        Tier 2 — PostgreSQL episodic memory
│   ├── semantic.py        Tier 3 — pgvector semantic memory
│   ├── retriever.py        Unified orchestration layer (the ONLY
│   │                       module agents should import from)
│   └── integration.py      Wiring into Planner/Researcher/Critic agents
├── api/
│   └── memory_endpoints.py FastAPI routes backing the dashboard
├── db/
│   └── 003_memory_tiers.sql Migration — run this against your DB
├── tests/memory/
│   ├── conftest.py
│   ├── test_working_memory.py    18 tests — concurrency, TTL, bounds
│   ├── test_episodic_memory.py   16 tests — writes, supersession, pruning
│   └── test_semantic_memory.py   19 tests — consolidation, decay, conflicts
└── requirements-memory.txt

frontend/
└── src/components/
    └── MemoryPanel.tsx      Live dashboard — stats, conflicts, retrieval preview
```

## Setup

### 1. Database migration

```bash
psql $DATABASE_URL -f backend/db/003_memory_tiers.sql
```

### 2. Install dependencies

```bash
pip install -r backend/requirements-memory.txt --break-system-packages
```

### 3. Wire the router into your FastAPI app

```python
# backend/main.py
from app.api.memory_endpoints import router as memory_router
app.include_router(memory_router, prefix="/api/v1")
```

### 4. Mix memory into your existing agents

```python
# backend/agents.py
from app.memory.integration import (
    MemoryAwarePlannerMixin,
    MemoryAwareResearcherMixin,
    MemoryAwareCriticMixin,
)

class PlannerAgent(MemoryAwarePlannerMixin, BaseAgent):
    async def _execute(self, state):
        memory_context = await self.retrieve_memory_context(
            query=state.query, memory=self.memory, domain=state.domain
        )
        prompt = self.build_planning_prompt_with_memory(base_prompt, memory_context)
        # ... rest of planning as before

class ResearcherAgent(MemoryAwareResearcherMixin, BaseAgent):
    async def _execute(self, state):
        # ... existing research logic
        await self.write_through_working_memory(
            self.working_memory, self.agent_id, finding_text, confidence
        )
```

### 5. Finalize + consolidate at pipeline completion

```python
# backend/graph/pipeline.py — at the end of ResearchPipeline.run()
from app.memory.integration import finalize_and_consolidate

await finalize_and_consolidate(
    memory=self.memory_retriever,
    session_id=state.session_id,
    query=state.query,
    user_id=state.user_id,
    domain=state.domain,
    status="complete",
    final_confidence=state.report.overall_confidence,
    topics=[s.heading for s in state.report.sections],
    settled_beliefs={...},
    contradictions_found=state.metadata.get("contradiction_count", 0),
    contradictions_resolved=state.metadata.get("resolved_count", 0),
    source_urls=[s.url for s in state.report.all_sources],
    avg_source_trust=avg_trust,
    total_tokens=self.budget.used,
    duration_seconds=elapsed,
    refinement_iterations=state.refinement_count,
    report_summary=state.report.executive_summary,
    full_report_json=state.report.model_dump(mode="json"),
)
```

### 6. Mount the frontend panel

```tsx
import { MemoryPanel } from "./components/MemoryPanel";

// Add a "Memory" tab alongside Agents/Debate/Beliefs/Benchmarks
{tab === "memory" && <MemoryPanel />}
```

## Running tests

```bash
# Working memory tests (fakeredis, no external deps)
pip install fakeredis --break-system-packages
pytest backend/tests/memory/test_working_memory.py -v

# Episodic + semantic tests (need real PostgreSQL + pgvector)
docker run -d -p 5432:5432 \
  -e POSTGRES_USER=mars_test -e POSTGRES_PASSWORD=mars_test \
  -e POSTGRES_DB=mars_test_db \
  pgvector/pgvector:pg16

pytest backend/tests/memory/ -v
```

53 tests total across the three tiers.

## Key production decisions

**Why Lua scripts for confidence updates?**
Two agents can finish within milliseconds of each other and both try to
update the same topic's confidence. A naive `GET` then `SET` has a race
window. Redis Lua scripts execute atomically — no race condition possible,
without needing distributed locks.

**Why is consolidation async, not inline?**
Fact extraction requires an LLM call per session, and contradiction
checking requires another LLM call per extracted fact. That's potentially
5-10 seconds of latency the user should never wait for after their report
is ready. `asyncio.create_task()` fires it off; the user sees "complete"
immediately.

**Why HNSW at m=16, ef_construction=96 specifically?**
Per pgvector production benchmarking: `m=16-24, ef_construction=96-128` is
the sweet spot for embeddings under 1024 dimensions (we use 384-dim
all-MiniLM-L6-v2). Higher `m` increases recall but costs roughly
`4*m*N*1.1` bytes in graph edges alone — at `m=32` and 50M rows that's
already ~7GB just for the graph structure, before counting vectors.

**Why does confidence decay over time?**
A fact consolidated in 2024 saying "quantum fault tolerance is 8 years
away" shouldn't carry full weight in a 2026 query without being
re-verified. Exponential decay (90-day half-life, floored at 15% of
original) means stale facts fade in influence but never vanish entirely —
they can still surface as "3 years ago, MARS concluded X" context.

**Why do contradictions get flagged instead of resolved?**
Semantic memory is not the authority on truth — the Critic agent is.
Auto-resolving a contradiction (e.g. "newer wins") would silently discard
information that might actually be correct. Flagging preserves both
claims and routes the decision to the part of the system designed to
adjudicate evidence.
