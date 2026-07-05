"""
╔══════════════════════════════════════════════════════════════════════╗
║  EPISODIC MEMORY — Tier 2                                            ║
║                                                                      ║
║  Scope: persistent record of past research sessions. PostgreSQL.     ║
║                                                                      ║
║  When a pipeline completes, its full outcome is written here:        ║
║  the query, the settled beliefs, the contradictions found, the       ║
║  final confidence scores, and which sources mattered most.           ║
║                                                                      ║
║  This is "what happened" memory — episodic in the cognitive-science   ║
║  sense (Tulving, 1972): specific past events, not general knowledge.  ║
║  Semantic memory (tier 3) later distills these episodes into          ║
║  durable, cross-session facts.                                       ║
║                                                                      ║
║  Design constraints that matter in production:                       ║
║    - Append-only for the audit trail: episodes are never mutated     ║
║      after write, only superseded by newer episodes referencing them ║
║    - Indexed for two access patterns: by user (their research         ║
║      history) and by topic similarity (semantic search hands off      ║
║      to tier 3, but episodic supports exact/fuzzy topic lookup)       ║
║    - Bounded retention: episodes older than a configurable window     ║
║      are pruned (or archived to cold storage) — this is NOT           ║
║      unbounded accumulation                                          ║
║    - Every write is transactional — a session that failed midway      ║
║      must not leave a corrupt partial episode                        ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import structlog
from sqlalchemy import select, delete, func, and_, or_, Column, String, Float, Integer, DateTime, Text, ForeignKey
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base, relationship

log = structlog.get_logger(__name__)

Base = declarative_base()

DEFAULT_RETENTION_DAYS = 180        # Episodes older than this are eligible for pruning
MAX_EPISODES_PER_QUERY = 20         # Cap on history queries — prevents unbounded reads


# ── ORM models ─────────────────────────────────────────────────────────────────

class EpisodeRecord(Base):
    """
    One completed (or failed) research session, permanently recorded.

    This table is the ground truth for "what did MARS conclude, and when."
    Every row corresponds to exactly one pipeline run.
    """
    __tablename__ = "episodic_memory"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    session_id = Column(PGUUID(as_uuid=True), nullable=False, unique=True, index=True)
    user_id = Column(PGUUID(as_uuid=True), nullable=True, index=True)

    query = Column(Text, nullable=False)
    query_domain = Column(String(64), nullable=True, index=True)   # e.g. "technology", "medical"

    status = Column(String(32), nullable=False, default="complete")  # complete | failed | partial

    # Settled outcome
    final_confidence = Column(Float, nullable=True)
    topics = Column(ARRAY(String), default=list)
    settled_beliefs = Column(JSONB, default=dict)     # {topic: {claim, confidence}}
    contradictions_found = Column(Integer, default=0)
    contradictions_resolved = Column(Integer, default=0)

    # Source provenance
    source_urls = Column(ARRAY(String), default=list)
    source_count = Column(Integer, default=0)
    avg_source_trust = Column(Float, nullable=True)

    # Cost / performance
    total_tokens = Column(Integer, default=0)
    duration_seconds = Column(Float, nullable=True)
    refinement_iterations = Column(Integer, default=0)

    # Full report (for retrieval/replay — not queried directly, just archived)
    report_summary = Column(Text, nullable=True)
    full_report_json = Column(JSONB, nullable=True)

    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    superseded_by = Column(PGUUID(as_uuid=True), nullable=True)   # Points to a newer episode on same topic

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self.id),
            "session_id": str(self.session_id),
            "query": self.query,
            "query_domain": self.query_domain,
            "status": self.status,
            "final_confidence": self.final_confidence,
            "topics": self.topics or [],
            "settled_beliefs": self.settled_beliefs or {},
            "contradictions_found": self.contradictions_found,
            "contradictions_resolved": self.contradictions_resolved,
            "source_count": self.source_count,
            "avg_source_trust": self.avg_source_trust,
            "total_tokens": self.total_tokens,
            "duration_seconds": self.duration_seconds,
            "report_summary": self.report_summary,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "is_superseded": self.superseded_by is not None,
        }


class EpisodeTopicIndex(Base):
    """
    Denormalized index for fast topic-based episode lookup without a full
    semantic search. This is the "did I research something like this
    recently" fast path — a cheap exact/prefix match before falling back
    to the expensive pgvector similarity search in semantic memory.
    """
    __tablename__ = "episodic_topic_index"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    episode_id = Column(PGUUID(as_uuid=True), ForeignKey("episodic_memory.id", ondelete="CASCADE"), nullable=False)
    topic_normalized = Column(String(256), nullable=False, index=True)  # lowercased, stripped
    confidence = Column(Float, nullable=True)


# ── Data transfer objects ──────────────────────────────────────────────────────

@dataclass
class EpisodeWriteRequest:
    """Everything needed to record a completed pipeline as an episode."""
    session_id: UUID
    query: str
    user_id: UUID | None = None
    query_domain: str = "general"
    status: str = "complete"
    final_confidence: float | None = None
    topics: list[str] = field(default_factory=list)
    settled_beliefs: dict[str, Any] = field(default_factory=dict)
    contradictions_found: int = 0
    contradictions_resolved: int = 0
    source_urls: list[str] = field(default_factory=list)
    avg_source_trust: float | None = None
    total_tokens: int = 0
    duration_seconds: float | None = None
    refinement_iterations: int = 0
    report_summary: str | None = None
    full_report_json: dict[str, Any] | None = None
    error_message: str | None = None


@dataclass
class EpisodeQueryResult:
    episodes: list[dict[str, Any]]
    total_matching: int
    query_topic: str | None = None


class EpisodicMemoryError(Exception):
    """Raised when an episodic memory write or read fails."""


class EpisodicMemoryStore:
    """
    PostgreSQL-backed episodic memory. One instance is stateless and safe
    to construct per-request — it holds no connection state itself, only
    a reference to the async session factory.
    """

    def __init__(self, db_session: AsyncSession):
        self.db = db_session

    # ── Write path ───────────────────────────────────────────────────────────

    async def record_episode(self, request: EpisodeWriteRequest) -> EpisodeRecord:
        """
        Persist a completed pipeline run. Transactional: if topic-index
        insertion fails, the whole write rolls back — we never want an
        episode without its corresponding topic index entries, since that
        would make it invisible to fast-path lookup while still consuming
        storage.
        """
        episode = EpisodeRecord(
            session_id=request.session_id,
            user_id=request.user_id,
            query=request.query,
            query_domain=request.query_domain,
            status=request.status,
            final_confidence=request.final_confidence,
            topics=request.topics,
            settled_beliefs=request.settled_beliefs,
            contradictions_found=request.contradictions_found,
            contradictions_resolved=request.contradictions_resolved,
            source_urls=request.source_urls[:100],   # Bound array size defensively
            source_count=len(request.source_urls),
            avg_source_trust=request.avg_source_trust,
            total_tokens=request.total_tokens,
            duration_seconds=request.duration_seconds,
            refinement_iterations=request.refinement_iterations,
            report_summary=request.report_summary,
            full_report_json=request.full_report_json,
            error_message=request.error_message,
        )

        try:
            self.db.add(episode)
            await self.db.flush()   # Get the generated ID before adding topic index rows

            for topic in request.topics:
                topic_confidence = None
                belief = request.settled_beliefs.get(topic, {})
                if isinstance(belief, dict):
                    topic_confidence = belief.get("confidence")

                self.db.add(EpisodeTopicIndex(
                    episode_id=episode.id,
                    topic_normalized=topic.strip().lower()[:256],
                    confidence=topic_confidence,
                ))

            await self._mark_superseded_episodes(request.topics, episode.id)
            await self.db.commit()

        except Exception as exc:
            await self.db.rollback()
            log.error("episodic_memory.write_failed",
                     session_id=str(request.session_id), error=str(exc))
            raise EpisodicMemoryError(f"Failed to record episode: {exc}") from exc

        log.info("episodic_memory.recorded",
                 session_id=str(request.session_id),
                 topics=request.topics,
                 confidence=request.final_confidence)
        return episode

    async def _mark_superseded_episodes(
        self, new_topics: list[str], new_episode_id: UUID
    ) -> None:
        """
        When a new episode covers the same topics as older ones, mark the
        older episodes as superseded. This doesn't delete history (episodic
        memory is append-only) — it just tells semantic consolidation and
        retrieval to prefer the newer conclusion when topics collide.
        """
        if not new_topics:
            return

        normalized = [t.strip().lower() for t in new_topics]

        stmt = (
            select(EpisodeTopicIndex.episode_id)
            .where(EpisodeTopicIndex.topic_normalized.in_(normalized))
            .distinct()
        )
        result = await self.db.execute(stmt)
        older_episode_ids = [row[0] for row in result.all() if row[0] != new_episode_id]

        if older_episode_ids:
            update_stmt = (
                select(EpisodeRecord)
                .where(EpisodeRecord.id.in_(older_episode_ids))
                .where(EpisodeRecord.superseded_by.is_(None))
            )
            older_episodes = (await self.db.execute(update_stmt)).scalars().all()
            for old_ep in older_episodes:
                old_ep.superseded_by = new_episode_id

    # ── Read path ────────────────────────────────────────────────────────────

    async def get_by_session_id(self, session_id: UUID) -> EpisodeRecord | None:
        stmt = select(EpisodeRecord).where(EpisodeRecord.session_id == session_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_user_history(
        self,
        user_id: UUID,
        limit: int = MAX_EPISODES_PER_QUERY,
        include_superseded: bool = False,
    ) -> list[dict[str, Any]]:
        """Recent research history for a user, newest first."""
        limit = min(limit, MAX_EPISODES_PER_QUERY)

        stmt = (
            select(EpisodeRecord)
            .where(EpisodeRecord.user_id == user_id)
            .order_by(EpisodeRecord.created_at.desc())
            .limit(limit)
        )
        if not include_superseded:
            stmt = stmt.where(EpisodeRecord.superseded_by.is_(None))

        result = await self.db.execute(stmt)
        episodes = result.scalars().all()
        return [ep.to_dict() for ep in episodes]

    async def find_by_topic(
        self,
        topic: str,
        user_id: UUID | None = None,
        limit: int = 5,
        fuzzy: bool = True,
    ) -> EpisodeQueryResult:
        """
        Fast-path topic lookup — exact/prefix match on the normalized topic
        index. This is checked BEFORE falling back to expensive semantic
        (vector) search. Most repeat queries hit this path.
        """
        normalized = topic.strip().lower()

        conditions = [EpisodeTopicIndex.topic_normalized == normalized]
        if fuzzy:
            conditions.append(EpisodeTopicIndex.topic_normalized.ilike(f"%{normalized}%"))

        stmt = (
            select(EpisodeRecord)
            .join(EpisodeTopicIndex, EpisodeTopicIndex.episode_id == EpisodeRecord.id)
            .where(or_(*conditions))
            .where(EpisodeRecord.superseded_by.is_(None))
            .order_by(EpisodeRecord.created_at.desc())
        )
        if user_id is not None:
            stmt = stmt.where(EpisodeRecord.user_id == user_id)
        stmt = stmt.limit(limit)

        result = await self.db.execute(stmt)
        episodes = result.scalars().unique().all()

        return EpisodeQueryResult(
            episodes=[ep.to_dict() for ep in episodes],
            total_matching=len(episodes),
            query_topic=topic,
        )

    async def get_recent_by_domain(
        self, domain: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """All recent episodes in a domain — used to seed domain-specialist
        agent priors (e.g. 'what has MARS previously found about medical
        research topics')."""
        stmt = (
            select(EpisodeRecord)
            .where(EpisodeRecord.query_domain == domain)
            .where(EpisodeRecord.superseded_by.is_(None))
            .where(EpisodeRecord.status == "complete")
            .order_by(EpisodeRecord.created_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return [ep.to_dict() for ep in result.scalars().all()]

    # ── Aggregate stats (for eval dashboard) ────────────────────────────────

    async def get_stats(self, since: datetime | None = None) -> dict[str, Any]:
        since = since or (datetime.now(timezone.utc) - timedelta(days=30))

        stmt = select(
            func.count(EpisodeRecord.id),
            func.avg(EpisodeRecord.final_confidence),
            func.avg(EpisodeRecord.duration_seconds),
            func.sum(EpisodeRecord.total_tokens),
            func.avg(EpisodeRecord.contradictions_found),
        ).where(EpisodeRecord.created_at >= since)

        result = await self.db.execute(stmt)
        row = result.one()

        return {
            "period_start": since.isoformat(),
            "total_episodes": row[0] or 0,
            "avg_confidence": round(float(row[1]), 4) if row[1] else None,
            "avg_duration_seconds": round(float(row[2]), 2) if row[2] else None,
            "total_tokens_consumed": int(row[3] or 0),
            "avg_contradictions_per_session": round(float(row[4]), 2) if row[4] else None,
        }

    # ── Retention / pruning ──────────────────────────────────────────────────

    async def prune_expired(
        self, retention_days: int = DEFAULT_RETENTION_DAYS
    ) -> int:
        """
        Delete episodes older than the retention window. Called by a
        scheduled job (Celery beat), never inline in the request path.

        Only prunes SUPERSEDED episodes past retention — the latest episode
        on any given topic is kept indefinitely as the "current understanding."
        This means retention bounds storage growth for repeat-topic churn
        while never silently losing the most recent conclusion on anything.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=retention_days)

        stmt = delete(EpisodeRecord).where(
            and_(
                EpisodeRecord.created_at < cutoff,
                EpisodeRecord.superseded_by.isnot(None),
            )
        )
        result = await self.db.execute(stmt)
        await self.db.commit()

        deleted_count = result.rowcount
        log.info("episodic_memory.pruned",
                 deleted_count=deleted_count,
                 retention_days=retention_days,
                 cutoff=cutoff.isoformat())
        return deleted_count
