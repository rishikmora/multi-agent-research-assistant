"""
Pydantic v2 schemas — the contract between frontend, API, and agents.
All pipeline state is typed end-to-end.
"""
from __future__ import annotations
from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, HttpUrl, field_validator


# ── Enums ────────────────────────────────────────────────────────────────────

class AgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    SKIPPED = "skipped"


class ResearchStatus(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"
    RESEARCHING = "researching"
    CRITIQUING = "critiquing"
    REFINING = "refining"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    FAILED = "failed"


class SourceType(str, Enum):
    WEB = "web"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    PDF = "pdf"
    INTERNAL = "internal"


# ── Source & Citation ─────────────────────────────────────────────────────────

class Source(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    url: str
    title: str
    snippet: str = ""
    source_type: SourceType = SourceType.WEB
    published_date: datetime | None = None
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    authors: list[str] = []
    citation_count: int = 0

    @field_validator("credibility_score")
    @classmethod
    def round_score(cls, v: float) -> float:
        return round(v, 3)


class Citation(BaseModel):
    source_id: UUID
    claim: str
    quote: str = ""
    page_number: int | None = None


# ── Sub-task ──────────────────────────────────────────────────────────────────

class SubTask(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    heading: str
    objective: str
    assigned_to: str  # "researcher_a" | "researcher_b"
    allowed_sources: list[SourceType] = []
    scope_rules: list[str] = []
    priority: int = Field(default=1, ge=1, le=5)
    status: AgentStatus = AgentStatus.PENDING
    findings: list[str] = []
    sources: list[Source] = []


# ── Report sections ───────────────────────────────────────────────────────────

class ReportSection(BaseModel):
    heading: str
    content: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[Citation] = []
    word_count: int = 0
    gaps_noted: list[str] = []


class ResearchReport(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str
    executive_summary: str
    sections: list[ReportSection]
    all_sources: list[Source]
    overall_confidence: float = Field(ge=0.0, le=1.0)
    total_sources: int = 0
    word_count: int = 0
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    refinement_iterations: int = 0
    model_versions: dict[str, str] = {}


# ── Pipeline state (persisted in Redis + DB) ──────────────────────────────────

class AgentTrace(BaseModel):
    agent_id: str
    status: AgentStatus
    started_at: datetime | None = None
    completed_at: datetime | None = None
    duration_ms: int | None = None
    token_usage: dict[str, int] = {}
    tool_calls: list[dict[str, Any]] = []
    error: str | None = None


class PipelineState(BaseModel):
    session_id: UUID = Field(default_factory=uuid4)
    user_id: UUID | None = None
    query: str
    status: ResearchStatus = ResearchStatus.QUEUED
    sub_tasks: list[SubTask] = []
    agent_traces: dict[str, AgentTrace] = {}
    refinement_count: int = 0
    report: ResearchReport | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: str | None = None
    metadata: dict[str, Any] = {}


# ── SSE event envelope ────────────────────────────────────────────────────────

class SSEEventType(str, Enum):
    PIPELINE_START = "pipeline_start"
    AGENT_START = "agent_start"
    AGENT_PROGRESS = "agent_progress"
    AGENT_COMPLETE = "agent_complete"
    AGENT_ERROR = "agent_error"
    SOURCES_FOUND = "sources_found"
    REFINEMENT_LOOP = "refinement_loop"
    REPORT_SECTION = "report_section"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_ERROR = "pipeline_error"
    HEARTBEAT = "heartbeat"


class SSEEvent(BaseModel):
    event: SSEEventType
    session_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: dict[str, Any] = {}

    def to_sse_string(self) -> str:
        import json
        payload = self.model_dump_json()
        return f"event: {self.event.value}\ndata: {payload}\n\n"


# ── API request/response ──────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=10, max_length=2000)
    depth: Literal["quick", "standard", "deep"] = "standard"  # type: ignore[name-defined]
    max_sources: int = Field(default=20, ge=5, le=50)
    include_arxiv: bool = True
    language: str = "en"


class ResearchResponse(BaseModel):
    session_id: str
    status: ResearchStatus
    message: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    checks: dict[str, bool] = {}


# fix forward ref
from typing import Literal  # noqa: E402 — after enum definitions
