-- Migration 003: Three-tier memory system
-- Tier 1 (working memory) lives entirely in Redis — no SQL needed.
-- This migration creates tier 2 (episodic) and tier 3 (semantic) tables.

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ══════════════════════════════════════════════════════════════════════════
-- TIER 2 — EPISODIC MEMORY
-- ══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS episodic_memory (
    id                       UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id               UUID NOT NULL UNIQUE,
    user_id                  UUID,

    query                    TEXT NOT NULL,
    query_domain             VARCHAR(64),

    status                   VARCHAR(32) NOT NULL DEFAULT 'complete',

    final_confidence         FLOAT,
    topics                   TEXT[] DEFAULT '{}',
    settled_beliefs          JSONB DEFAULT '{}',
    contradictions_found     INTEGER DEFAULT 0,
    contradictions_resolved  INTEGER DEFAULT 0,

    source_urls              TEXT[] DEFAULT '{}',
    source_count             INTEGER DEFAULT 0,
    avg_source_trust         FLOAT,

    total_tokens             INTEGER DEFAULT 0,
    duration_seconds         FLOAT,
    refinement_iterations    INTEGER DEFAULT 0,

    report_summary           TEXT,
    full_report_json         JSONB,

    error_message            TEXT,

    created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    superseded_by             UUID REFERENCES episodic_memory(id)
);

CREATE INDEX IF NOT EXISTS idx_episodic_session      ON episodic_memory(session_id);
CREATE INDEX IF NOT EXISTS idx_episodic_user          ON episodic_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_episodic_domain        ON episodic_memory(query_domain);
CREATE INDEX IF NOT EXISTS idx_episodic_created       ON episodic_memory(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_episodic_superseded    ON episodic_memory(superseded_by)
    WHERE superseded_by IS NOT NULL;

-- Fast-path topic index — checked before falling back to semantic vector search
CREATE TABLE IF NOT EXISTS episodic_topic_index (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    episode_id        UUID NOT NULL REFERENCES episodic_memory(id) ON DELETE CASCADE,
    topic_normalized  VARCHAR(256) NOT NULL,
    confidence        FLOAT
);

CREATE INDEX IF NOT EXISTS idx_topic_normalized ON episodic_topic_index(topic_normalized);
CREATE INDEX IF NOT EXISTS idx_topic_episode    ON episodic_topic_index(episode_id);

-- ══════════════════════════════════════════════════════════════════════════
-- TIER 3 — SEMANTIC MEMORY
-- ══════════════════════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS semantic_memory (
    id                     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    content                TEXT NOT NULL,
    content_hash           VARCHAR(32) NOT NULL UNIQUE,
    embedding              vector(384) NOT NULL,   -- all-MiniLM-L6-v2 dimension

    topics                 TEXT[] DEFAULT '{}',
    domain                 VARCHAR(64),

    confidence             FLOAT NOT NULL DEFAULT 0.6,
    corroboration_count    INTEGER DEFAULT 1,
    contradiction_count    INTEGER DEFAULT 0,

    source_episode_ids     UUID[] DEFAULT '{}',

    is_contested           BOOLEAN DEFAULT FALSE,
    contested_reason       TEXT,

    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_reinforced_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reinforcement_count    INTEGER DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_semantic_hash       ON semantic_memory(content_hash);
CREATE INDEX IF NOT EXISTS idx_semantic_domain      ON semantic_memory(domain);
CREATE INDEX IF NOT EXISTS idx_semantic_reinforced  ON semantic_memory(last_reinforced_at);
CREATE INDEX IF NOT EXISTS idx_semantic_contested   ON semantic_memory(is_contested)
    WHERE is_contested = TRUE;

-- Production-tuned HNSW index.
-- m=16, ef_construction=96: within the recommended production range
-- (m=16-24, ef_construction=96-128) for embeddings under 1024 dimensions.
-- At m=16 with N rows, HNSW graph edge storage is roughly 4*16*N*1.1 bytes —
-- for 100k semantic entries that's ~70MB of graph structure, well within
-- reasonable memory budgets even on modest instances.
CREATE INDEX IF NOT EXISTS idx_semantic_embedding
    ON semantic_memory
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 96);

-- Conflict tracking — surfaced to Critic agent, never silently auto-resolved
CREATE TABLE IF NOT EXISTS semantic_conflicts (
    id                      UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    semantic_entry_id       UUID REFERENCES semantic_memory(id) ON DELETE CASCADE,
    conflicting_episode_id  UUID NOT NULL,
    conflicting_claim       TEXT NOT NULL,
    existing_claim          TEXT NOT NULL,
    similarity_score        FLOAT,
    resolved                BOOLEAN DEFAULT FALSE,
    resolution              TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conflicts_unresolved ON semantic_conflicts(resolved)
    WHERE resolved = FALSE;
CREATE INDEX IF NOT EXISTS idx_conflicts_entry ON semantic_conflicts(semantic_entry_id);

-- ══════════════════════════════════════════════════════════════════════════
-- MAINTENANCE — scheduled pruning (called by Celery beat, not inline)
-- ══════════════════════════════════════════════════════════════════════════

-- Analytical view: memory system health at a glance
CREATE OR REPLACE VIEW memory_system_health AS
SELECT
    (SELECT COUNT(*) FROM episodic_memory) AS total_episodes,
    (SELECT COUNT(*) FROM episodic_memory WHERE superseded_by IS NULL) AS current_episodes,
    (SELECT AVG(final_confidence) FROM episodic_memory WHERE status = 'complete') AS avg_episode_confidence,
    (SELECT COUNT(*) FROM semantic_memory) AS total_semantic_facts,
    (SELECT AVG(confidence) FROM semantic_memory) AS avg_semantic_confidence,
    (SELECT AVG(corroboration_count) FROM semantic_memory) AS avg_corroboration,
    (SELECT COUNT(*) FROM semantic_memory WHERE is_contested = TRUE) AS contested_facts,
    (SELECT COUNT(*) FROM semantic_conflicts WHERE resolved = FALSE) AS pending_conflicts;

COMMENT ON TABLE episodic_memory IS
    'Tier 2: append-only record of every completed/failed research session. Never mutated, only superseded.';
COMMENT ON TABLE semantic_memory IS
    'Tier 3: durable cross-session knowledge, consolidated from episodic memory. Confidence decays over time without reinforcement.';
COMMENT ON COLUMN semantic_memory.embedding IS
    'all-MiniLM-L6-v2, 384-dim, normalized. HNSW index at m=16/ef_construction=96 (production tuning for <1024-dim vectors).';
