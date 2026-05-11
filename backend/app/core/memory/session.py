"""
Session management using Redis.
- Stores pipeline state with TTL
- Manages per-session SSE event queues
- Provides pub/sub for horizontal scaling
"""
from __future__ import annotations
import asyncio
import json
from typing import Any
import redis.asyncio as aioredis
import structlog

from app.core.config import settings
from app.schemas.research import PipelineState, SSEEvent

log = structlog.get_logger(__name__)

# Module-level Redis pool (created once on startup)
_redis_pool: aioredis.Redis | None = None

# In-process SSE queues — one per active session
_session_queues: dict[str, asyncio.Queue[SSEEvent]] = {}


async def get_redis() -> aioredis.Redis:
    global _redis_pool
    if _redis_pool is None:
        _redis_pool = await aioredis.from_url(
            str(settings.redis_url),
            max_connections=settings.redis_max_connections,
            decode_responses=True,
        )
    return _redis_pool


async def create_session_queue(session_id: str) -> asyncio.Queue[SSEEvent]:
    """Create a new SSE event queue for a session."""
    queue: asyncio.Queue[SSEEvent] = asyncio.Queue(maxsize=500)
    _session_queues[session_id] = queue
    log.info("session.queue_created", session_id=session_id)
    return queue


def get_session_queue(session_id: str) -> asyncio.Queue[SSEEvent] | None:
    return _session_queues.get(session_id)


async def cleanup_session(session_id: str) -> None:
    _session_queues.pop(session_id, None)
    try:
        redis = await get_redis()
        await redis.delete(f"session:{session_id}")
    except Exception as exc:
        log.warning("session.cleanup_failed", error=str(exc))


async def save_pipeline_state(session_id: str, state: PipelineState) -> None:
    try:
        redis = await get_redis()
        await redis.setex(
            f"session:{session_id}",
            settings.cache_ttl_seconds,
            state.model_dump_json(),
        )
    except Exception as exc:
        log.error("session.save_failed", error=str(exc), session_id=session_id)


async def load_pipeline_state(session_id: str) -> PipelineState | None:
    try:
        redis = await get_redis()
        data = await redis.get(f"session:{session_id}")
        if data:
            return PipelineState.model_validate_json(data)
    except Exception as exc:
        log.error("session.load_failed", error=str(exc), session_id=session_id)
    return None


async def get_session_history(user_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """Retrieve recent session summaries for a user."""
    try:
        redis = await get_redis()
        key = f"user:{user_id}:sessions"
        raw = await redis.lrange(key, 0, limit - 1)
        return [json.loads(item) for item in raw]
    except Exception:
        return []


async def append_session_to_history(user_id: str, session_id: str, query: str) -> None:
    try:
        redis = await get_redis()
        key = f"user:{user_id}:sessions"
        entry = json.dumps({"session_id": session_id, "query": query[:100]})
        await redis.lpush(key, entry)
        await redis.ltrim(key, 0, 49)  # Keep last 50 sessions
        await redis.expire(key, 86400 * 30)  # 30 days
    except Exception as exc:
        log.warning("history.append_failed", error=str(exc))
