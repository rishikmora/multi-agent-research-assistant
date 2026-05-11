-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Research sessions table
CREATE TABLE IF NOT EXISTS research_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID,
    query TEXT NOT NULL,
    status VARCHAR(32) NOT NULL DEFAULT 'queued',
    report JSONB,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    error_message TEXT,
    token_usage INTEGER DEFAULT 0,
    refinement_count INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON research_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON research_sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON research_sessions(created_at DESC);

-- Source documents table (deduplicated across sessions)
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    url TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    snippet TEXT DEFAULT '',
    source_type VARCHAR(32) NOT NULL,
    content TEXT DEFAULT '',
    credibility_score FLOAT DEFAULT 0.5,
    published_date TIMESTAMPTZ,
    authors TEXT[] DEFAULT '{}',
    citation_count INTEGER DEFAULT 0,
    retrieved_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    -- pgvector embedding for semantic search
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url);
CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type);
-- HNSW index for fast approximate nearest-neighbor search
CREATE INDEX IF NOT EXISTS idx_sources_embedding ON sources
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Session-source junction
CREATE TABLE IF NOT EXISTS session_sources (
    session_id UUID REFERENCES research_sessions(id) ON DELETE CASCADE,
    source_id UUID REFERENCES sources(id) ON DELETE CASCADE,
    PRIMARY KEY (session_id, source_id)
);

-- LangGraph checkpoints (managed by LangGraph library)
CREATE TABLE IF NOT EXISTS checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    type TEXT,
    checkpoint JSONB NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_id)
);

CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints(thread_id);

-- Semantic memory (mem0 / long-term per-user memory)
CREATE TABLE IF NOT EXISTS memory_vectors (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_memory_user ON memory_vectors(user_id);
CREATE INDEX IF NOT EXISTS idx_memory_embedding ON memory_vectors
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Trigger to auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = NOW(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER sessions_updated_at
    BEFORE UPDATE ON research_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
