"""
╔══════════════════════════════════════════════════════════════════════╗
║  WORKING MEMORY — Tier 1                                             ║
║                                                                      ║
║  Scope: current pipeline session only. TTL-bound. Redis-backed.      ║
║                                                                      ║
║  This is the "scratchpad" every agent reads and writes during a      ║
║  single research run. It holds:                                     ║
║    - Every agent's raw output as it completes                        ║
║    - Running confidence scores per topic                             ║
║    - The evidence graph state (nodes/edges) as it's built live       ║
║    - Debate transcript as rounds complete                            ║
║                                                                      ║
║  Design constraints that matter in production:                       ║
║    - Atomic writes: two agents finishing concurrently must not       ║
║      clobber each other's data (Redis WATCH/MULTI or Lua script)     ║
║    - Bounded size: a session's working memory must not grow          ║
║      unbounded — hard cap enforced, oldest entries evicted           ║
║    - TTL: sessions expire automatically (default 2 hours) so a       ║
║      crashed pipeline doesn't leak memory forever                    ║
║    - Pub/sub: other processes (SSE handler, eval worker) can         ║
║      subscribe to session updates without polling                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import redis.asyncio as aioredis
import structlog

from app.core.config import settings

log = structlog.get_logger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

WORKING_MEMORY_TTL_SECONDS = 7200          # 2 hours — auto-expire crashed sessions
MAX_AGENT_OUTPUTS_PER_SESSION = 50          # Hard cap — prevents unbounded growth
MAX_DEBATE_MESSAGES = 30                    # Hard cap on debate transcript
KEY_PREFIX = "wm"                           # working-memory key namespace


class WorkingMemoryError(Exception):
    """Raised when a working memory operation fails after retries."""


class EntryType(str, Enum):
    AGENT_OUTPUT = "agent_output"
    CONFIDENCE_UPDATE = "confidence_update"
    DEBATE_MESSAGE = "debate_message"
    GRAPH_MUTATION = "graph_mutation"
    METADATA = "metadata"


@dataclass
class AgentOutputEntry:
    """One agent's completed output, stored in working memory."""
    agent_id: str
    output_text: str
    confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    token_usage: dict[str, int] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionSnapshot:
    """Full working-memory state for a session, used for SSE hydration."""
    session_id: str
    query: str
    status: str
    agent_outputs: dict[str, AgentOutputEntry]
    confidence_by_topic: dict[str, float]
    debate_transcript: list[dict[str, Any]]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "status": self.status,
            "agent_outputs": {k: v.to_dict() for k, v in self.agent_outputs.items()},
            "confidence_by_topic": self.confidence_by_topic,
            "debate_transcript": self.debate_transcript,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ── Redis key helpers ─────────────────────────────────────────────────────────

def _k_session(session_id: str) -> str:
    return f"{KEY_PREFIX}:session:{session_id}"

def _k_outputs(session_id: str) -> str:
    return f"{KEY_PREFIX}:outputs:{session_id}"

def _k_confidence(session_id: str) -> str:
    return f"{KEY_PREFIX}:confidence:{session_id}"

def _k_debate(session_id: str) -> str:
    return f"{KEY_PREFIX}:debate:{session_id}"

def _k_graph(session_id: str) -> str:
    return f"{KEY_PREFIX}:graph:{session_id}"

def _k_lock(session_id: str) -> str:
    return f"{KEY_PREFIX}:lock:{session_id}"

def _k_pubsub(session_id: str) -> str:
    return f"{KEY_PREFIX}:events:{session_id}"


# ── Lua script for atomic confidence updates ──────────────────────────────────
# Redis Lua scripts execute atomically — no race condition between
# read-modify-write when two agents update the same topic's confidence
# within microseconds of each other.

_ATOMIC_CONFIDENCE_UPDATE_SCRIPT = """
local key = KEYS[1]
local topic = ARGV[1]
local new_value = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

redis.call('HSET', key, topic, new_value)
redis.call('EXPIRE', key, ttl)
return new_value
"""

_BOUNDED_LIST_PUSH_SCRIPT = """
local key = KEYS[1]
local value = ARGV[1]
local max_len = tonumber(ARGV[2])
local ttl = tonumber(ARGV[3])

redis.call('RPUSH', key, value)
local len = redis.call('LLEN', key)
if len > max_len then
    redis.call('LPOP', key, len - max_len)
end
redis.call('EXPIRE', key, ttl)
return redis.call('LLEN', key)
"""


class WorkingMemory:
    """
    Redis-backed working memory for a single pipeline session.

    Usage:
        wm = WorkingMemory(session_id, redis_client)
        await wm.initialize(query="...")
        await wm.record_agent_output("researcher_a", "findings...", confidence=0.81)
        await wm.update_confidence("hardware breakthroughs", 0.76)
        snapshot = await wm.get_snapshot()
    """

    def __init__(self, session_id: str, redis: aioredis.Redis):
        self.session_id = session_id
        self.redis = redis
        self._confidence_script = None
        self._bounded_push_script = None

    async def _ensure_scripts_loaded(self) -> None:
        """Register Lua scripts with Redis on first use (cached by SHA)."""
        if self._confidence_script is None:
            self._confidence_script = self.redis.register_script(
                _ATOMIC_CONFIDENCE_UPDATE_SCRIPT
            )
        if self._bounded_push_script is None:
            self._bounded_push_script = self.redis.register_script(
                _BOUNDED_LIST_PUSH_SCRIPT
            )

    # ── Lifecycle ────────────────────────────────────────────────────────────

    async def initialize(self, query: str) -> None:
        """Create the session record. Idempotent — safe to call on retry."""
        now = datetime.now(timezone.utc).isoformat()
        session_data = {
            "session_id": self.session_id,
            "query": query,
            "status": "queued",
            "created_at": now,
            "updated_at": now,
        }
        await self.redis.hset(_k_session(self.session_id), mapping=session_data)
        await self.redis.expire(_k_session(self.session_id), WORKING_MEMORY_TTL_SECONDS)
        log.info("working_memory.initialized",
                 session_id=self.session_id,
                 query=query[:80])

    async def set_status(self, status: str) -> None:
        await self.redis.hset(_k_session(self.session_id), "status", status)
        await self.redis.hset(
            _k_session(self.session_id),
            "updated_at",
            datetime.now(timezone.utc).isoformat(),
        )
        await self._publish_event({"type": "status_change", "status": status})

    async def touch_ttl(self) -> None:
        """Refresh TTL on all session keys — call periodically during long runs."""
        pipe = self.redis.pipeline()
        for key_fn in (_k_session, _k_outputs, _k_confidence, _k_debate, _k_graph):
            pipe.expire(key_fn(self.session_id), WORKING_MEMORY_TTL_SECONDS)
        await pipe.execute()

    async def delete(self) -> None:
        """Explicit cleanup — called when a session completes and is persisted
        to episodic memory. Working memory is transient by design."""
        keys = [
            _k_session(self.session_id),
            _k_outputs(self.session_id),
            _k_confidence(self.session_id),
            _k_debate(self.session_id),
            _k_graph(self.session_id),
            _k_lock(self.session_id),
        ]
        await self.redis.delete(*keys)
        log.info("working_memory.deleted", session_id=self.session_id)

    # ── Agent outputs ────────────────────────────────────────────────────────

    async def record_agent_output(
        self,
        agent_id: str,
        output_text: str,
        confidence: float,
        token_usage: dict[str, int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """
        Store a completed agent's output. Uses HSET so concurrent agents
        writing different fields (different agent_id keys) never conflict —
        Redis hash field writes are independently atomic.
        """
        entry = AgentOutputEntry(
            agent_id=agent_id,
            output_text=output_text,
            confidence=confidence,
            token_usage=token_usage or {},
            metadata=metadata or {},
        )

        key = _k_outputs(self.session_id)
        current_count = await self.redis.hlen(key)
        if current_count >= MAX_AGENT_OUTPUTS_PER_SESSION:
            log.warning("working_memory.output_cap_reached",
                        session_id=self.session_id,
                        cap=MAX_AGENT_OUTPUTS_PER_SESSION)
            # Evict oldest by scanning — rare path, session is misbehaving
            oldest_field = await self._find_oldest_output_field(key)
            if oldest_field:
                await self.redis.hdel(key, oldest_field)

        await self.redis.hset(key, agent_id, json.dumps(entry.to_dict()))
        await self.redis.expire(key, WORKING_MEMORY_TTL_SECONDS)

        await self._publish_event({
            "type": "agent_output",
            "agent_id": agent_id,
            "confidence": confidence,
        })

        log.info("working_memory.output_recorded",
                 session_id=self.session_id,
                 agent_id=agent_id,
                 confidence=confidence,
                 output_len=len(output_text))

    async def _find_oldest_output_field(self, key: str) -> str | None:
        all_entries = await self.redis.hgetall(key)
        if not all_entries:
            return None
        parsed = [
            (field, json.loads(val).get("timestamp", ""))
            for field, val in all_entries.items()
        ]
        parsed.sort(key=lambda x: x[1])
        return parsed[0][0] if parsed else None

    async def get_agent_output(self, agent_id: str) -> AgentOutputEntry | None:
        raw = await self.redis.hget(_k_outputs(self.session_id), agent_id)
        if raw is None:
            return None
        data = json.loads(raw)
        return AgentOutputEntry(**data)

    async def get_all_agent_outputs(self) -> dict[str, AgentOutputEntry]:
        raw = await self.redis.hgetall(_k_outputs(self.session_id))
        return {
            agent_id: AgentOutputEntry(**json.loads(val))
            for agent_id, val in raw.items()
        }

    # ── Confidence tracking (atomic via Lua) ────────────────────────────────

    async def update_confidence(self, topic: str, confidence: float) -> float:
        """
        Atomically update confidence for a topic. Safe under concurrent
        writers — the Lua script executes as a single Redis operation,
        so there's no read-modify-write race between the Critic and a
        Researcher updating the same topic within the same millisecond.
        """
        await self._ensure_scripts_loaded()
        result = await self._confidence_script(
            keys=[_k_confidence(self.session_id)],
            args=[topic, str(confidence), str(WORKING_MEMORY_TTL_SECONDS)],
        )
        await self._publish_event({
            "type": "confidence_update",
            "topic": topic,
            "confidence": confidence,
        })
        return float(result)

    async def get_confidence(self, topic: str) -> float | None:
        val = await self.redis.hget(_k_confidence(self.session_id), topic)
        return float(val) if val is not None else None

    async def get_all_confidence(self) -> dict[str, float]:
        raw = await self.redis.hgetall(_k_confidence(self.session_id))
        return {topic: float(v) for topic, v in raw.items()}

    # ── Debate transcript (bounded list via Lua) ────────────────────────────

    async def append_debate_message(self, message: dict[str, Any]) -> int:
        """
        Append to the debate transcript with a hard bound. If the transcript
        exceeds MAX_DEBATE_MESSAGES, oldest entries are trimmed automatically —
        this is a safety valve against a misconfigured debate loop running
        away and consuming unbounded memory.
        """
        await self._ensure_scripts_loaded()
        new_len = await self._bounded_push_script(
            keys=[_k_debate(self.session_id)],
            args=[
                json.dumps(message),
                str(MAX_DEBATE_MESSAGES),
                str(WORKING_MEMORY_TTL_SECONDS),
            ],
        )
        await self._publish_event({"type": "debate_message", "round": message.get("round")})
        return int(new_len)

    async def get_debate_transcript(self) -> list[dict[str, Any]]:
        raw = await self.redis.lrange(_k_debate(self.session_id), 0, -1)
        return [json.loads(item) for item in raw]

    # ── Live evidence graph state ────────────────────────────────────────────

    async def set_graph_state(self, graph_json: dict[str, Any]) -> None:
        """
        Store the current serialized evidence graph. This is called after
        every graph mutation (node/edge added) so the SSE handler can push
        live graph updates to the frontend without querying PostgreSQL.
        """
        await self.redis.set(
            _k_graph(self.session_id),
            json.dumps(graph_json),
            ex=WORKING_MEMORY_TTL_SECONDS,
        )
        await self._publish_event({
            "type": "graph_mutation",
            "node_count": len(graph_json.get("nodes", [])),
            "edge_count": len(graph_json.get("edges", [])),
        })

    async def get_graph_state(self) -> dict[str, Any] | None:
        raw = await self.redis.get(_k_graph(self.session_id))
        return json.loads(raw) if raw else None

    # ── Full snapshot (for SSE reconnection / late subscribers) ─────────────

    async def get_snapshot(self) -> SessionSnapshot | None:
        session_data = await self.redis.hgetall(_k_session(self.session_id))
        if not session_data:
            return None

        outputs = await self.get_all_agent_outputs()
        confidence = await self.get_all_confidence()
        debate = await self.get_debate_transcript()

        return SessionSnapshot(
            session_id=session_data["session_id"],
            query=session_data["query"],
            status=session_data["status"],
            agent_outputs=outputs,
            confidence_by_topic=confidence,
            debate_transcript=debate,
            created_at=session_data["created_at"],
            updated_at=session_data["updated_at"],
        )

    # ── Pub/sub for live event streaming ─────────────────────────────────────

    async def _publish_event(self, event: dict[str, Any]) -> None:
        """
        Publish to the session's pub/sub channel. The SSE endpoint subscribes
        to this channel — this decouples the pipeline execution process from
        the HTTP-facing SSE process, which matters when running multiple
        backend workers (Gunicorn with >1 worker, or separate Celery workers).
        Without pub/sub, a background task in worker A can never notify an
        SSE connection being served by worker B.
        """
        try:
            await self.redis.publish(
                _k_pubsub(self.session_id),
                json.dumps({**event, "session_id": self.session_id, "ts": time.time()}),
            )
        except Exception as exc:
            # Pub/sub failures must never crash the pipeline — SSE is best-effort
            log.warning("working_memory.publish_failed",
                        session_id=self.session_id, error=str(exc))

    def subscribe_channel_name(self) -> str:
        """Channel name for external subscribers (SSE handler calls this)."""
        return _k_pubsub(self.session_id)


# ── Factory / connection management ───────────────────────────────────────────

_redis_pool: aioredis.ConnectionPool | None = None


def get_redis_pool() -> aioredis.ConnectionPool:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = aioredis.ConnectionPool.from_url(
            str(settings.redis_url),
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
    return _redis_pool


async def get_redis_client() -> aioredis.Redis:
    return aioredis.Redis(connection_pool=get_redis_pool())


async def create_working_memory(session_id: str) -> WorkingMemory:
    """Factory — the standard way to obtain a WorkingMemory instance."""
    redis = await get_redis_client()
    return WorkingMemory(session_id, redis)
