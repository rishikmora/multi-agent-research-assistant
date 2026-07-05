"""
╔══════════════════════════════════════════════════════════════════════╗
║  SEMANTIC MEMORY — Tier 3                                            ║
║                                                                      ║
║  Scope: durable, cross-session, cross-user knowledge. pgvector.      ║
║                                                                      ║
║  This is "how things work" memory — general facts distilled from     ║
║  many episodes, not any single event. When 5 different sessions      ║
║  independently conclude "IBM's Condor processor has 1,121 qubits,"   ║
║  that fact gets consolidated into ONE semantic memory entry with      ║
║  boosted confidence — rather than living as 5 separate episodic       ║
║  records that a future session would have to re-discover.            ║
║                                                                      ║
║  The consolidation mechanism (episodic → semantic) is the primary    ║
║  driver of "lifelong learning" in multi-agent systems per current     ║
║  memory-architecture research: as the system operates, semantic       ║
║  memory grows by folding in newly-settled episodic conclusions.       ║
║                                                                      ║
║  Design constraints that matter in production:                       ║
║    - HNSW index tuned for 384-dim embeddings at m=16, ef_construction ║
║      =96 — production values that balance recall against build time  ║
║      and memory footprint (per pgvector benchmarking guidance:        ║
║      m=16-24, ef_construction=96-128 is the production sweet spot     ║
║      for sub-1024-dim vectors; going past m=32 costs ~4*m*N*1.1       ║
║      bytes in graph edges alone)                                      ║
║    - Consolidation is idempotent: re-running consolidation on the      ║
║      same episodes must not create duplicate semantic entries          ║
║    - Confidence decay: semantic facts that haven't been reinforced     ║
║      by new episodes lose confidence over time (staleness matters)     ║
║    - Contradiction handling: a new episode that CONTRADICTS an          ║
║      existing semantic fact doesn't silently overwrite it — it         ║
║      creates a flagged conflict for the Critic to resolve               ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import structlog
from pgvector.sqlalchemy import Vector
from sqlalchemy import select, delete, update, func, and_, text, Column, String, Float, Integer, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.dialects.postgresql import JSONB, ARRAY, UUID as PGUUID
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import declarative_base

log = structlog.get_logger(__name__)

Base = declarative_base()

EMBEDDING_DIM = 384                    # all-MiniLM-L6-v2 output dimension
HNSW_M = 16                            # Production value: m=16-24 for <1024 dims
HNSW_EF_CONSTRUCTION = 96              # Production value: 96-128 for good recall/build tradeoff
DEFAULT_SIMILARITY_THRESHOLD = 0.75    # Cosine similarity floor for "relevant"
CONFIDENCE_DECAY_HALF_LIFE_DAYS = 90   # Semantic facts halve in confidence every 90d if unreinforced
CONSOLIDATION_MIN_CORROBORATION = 2    # Need 2+ independent episodes to consolidate a fact


# ── ORM models ─────────────────────────────────────────────────────────────────

class SemanticMemoryEntry(Base):
    """
    A single unit of durable, cross-session knowledge.

    Each row represents one distilled fact/conclusion, backed by an
    embedding for similarity retrieval and a corroboration count that
    tracks how many independent episodes support it.
    """
    __tablename__ = "semantic_memory"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)

    content = Column(Text, nullable=False)              # The distilled fact, in natural language
    content_hash = Column(String(32), nullable=False, unique=True, index=True)  # Dedup key
    embedding = Column(Vector(EMBEDDING_DIM), nullable=False)

    topics = Column(ARRAY(String), default=list, index=False)
    domain = Column(String(64), nullable=True, index=True)

    confidence = Column(Float, nullable=False, default=0.6)
    corroboration_count = Column(Integer, default=1)      # How many episodes support this
    contradiction_count = Column(Integer, default=0)       # How many episodes conflict with this

    source_episode_ids = Column(ARRAY(PGUUID(as_uuid=True)), default=list)

    is_contested = Column(Boolean, default=False)          # Flagged for Critic review
    contested_reason = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_reinforced_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    reinforcement_count = Column(Integer, default=1)

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        d = {
            "id": str(self.id),
            "content": self.content,
            "topics": self.topics or [],
            "domain": self.domain,
            "confidence": round(self.confidence, 4),
            "corroboration_count": self.corroboration_count,
            "contradiction_count": self.contradiction_count,
            "is_contested": self.is_contested,
            "contested_reason": self.contested_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_reinforced_at": self.last_reinforced_at.isoformat() if self.last_reinforced_at else None,
            "reinforcement_count": self.reinforcement_count,
            "source_episode_count": len(self.source_episode_ids or []),
        }
        if include_embedding:
            d["embedding"] = list(self.embedding) if self.embedding is not None else None
        return d


class SemanticConflict(Base):
    """
    Records a detected contradiction between a new episode's conclusion
    and existing semantic memory. This table feeds the Critic agent's
    contradiction-resolution queue — semantic memory itself never silently
    resolves conflicts, it surfaces them.
    """
    __tablename__ = "semantic_conflicts"

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    semantic_entry_id = Column(PGUUID(as_uuid=True), ForeignKey("semantic_memory.id", ondelete="CASCADE"))
    conflicting_episode_id = Column(PGUUID(as_uuid=True), nullable=False)
    conflicting_claim = Column(Text, nullable=False)
    existing_claim = Column(Text, nullable=False)
    similarity_score = Column(Float, nullable=True)
    resolved = Column(Boolean, default=False)
    resolution = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


# ── DDL for HNSW index (run once via migration) ───────────────────────────────

HNSW_INDEX_DDL = f"""
CREATE INDEX IF NOT EXISTS idx_semantic_memory_embedding
ON semantic_memory
USING hnsw (embedding vector_cosine_ops)
WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
"""

# At query time, ef_search controls the recall/latency tradeoff for THIS query.
# Higher = better recall, more latency. 40 is a reasonable default for
# sub-100ms lookups against a corpus in the tens-of-thousands of rows.
SET_EF_SEARCH_SQL = "SET hnsw.ef_search = :ef_search"


# ── Data transfer objects ──────────────────────────────────────────────────────

@dataclass
class SemanticSearchResult:
    entry: dict[str, Any]
    similarity: float


@dataclass
class ConsolidationCandidate:
    """One episode's conclusion, pending consolidation into semantic memory."""
    episode_id: UUID
    content: str
    embedding: list[float]
    topics: list[str]
    domain: str
    confidence: float


class SemanticMemoryError(Exception):
    pass


def _content_hash(content: str) -> str:
    """Normalize and hash content for exact-dedup detection before
    falling back to embedding similarity for near-dup detection."""
    normalized = " ".join(content.strip().lower().split())
    return hashlib.md5(normalized.encode()).hexdigest()


def _decay_confidence(base_confidence: float, last_reinforced: datetime) -> float:
    """
    Exponential decay: confidence halves every CONFIDENCE_DECAY_HALF_LIFE_DAYS
    if the fact hasn't been reinforced by a new corroborating episode.
    A fact that's been sitting unreinforced for a year drops to near-zero
    influence — this prevents semantic memory from calcifying around
    outdated conclusions (e.g. "quantum computing is 10 years from
    commercial viability" stated in 2020 should not carry full weight
    in 2026 without reinforcement).
    """
    now = datetime.now(timezone.utc)
    if last_reinforced.tzinfo is None:
        last_reinforced = last_reinforced.replace(tzinfo=timezone.utc)
    days_stale = (now - last_reinforced).days
    if days_stale <= 0:
        return base_confidence
    decay_factor = 0.5 ** (days_stale / CONFIDENCE_DECAY_HALF_LIFE_DAYS)
    # Floor at 15% of original — a fact never fully disappears, it just
    # stops dominating retrieval rankings
    return max(base_confidence * decay_factor, base_confidence * 0.15)


class SemanticMemoryStore:
    """
    pgvector-backed semantic memory. Handles similarity search,
    consolidation from episodic memory, and contradiction detection.
    """

    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self._encoder = None

    def _get_encoder(self):
        """Lazy-loaded sentence-transformers model — avoids startup cost
        for requests that never touch semantic memory."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:
                log.error("semantic_memory.encoder_load_failed", error=str(exc))
                raise SemanticMemoryError(
                    "Embedding model unavailable — semantic memory degraded"
                ) from exc
        return self._encoder

    def embed(self, text_input: str) -> list[float]:
        encoder = self._get_encoder()
        vec = encoder.encode(text_input, normalize_embeddings=True)
        return vec.tolist()

    # ── Similarity search (the retrieval path) ──────────────────────────────

    async def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        domain: str | None = None,
        ef_search: int = 40,
        exclude_contested: bool = False,
    ) -> list[SemanticSearchResult]:
        """
        Find semantically relevant memory entries for a query.

        Sets hnsw.ef_search per-query (not globally) so concurrent requests
        with different latency/recall needs don't interfere with each other —
        this is a session-local SET, scoped to this transaction.
        """
        query_embedding = self.embed(query)

        await self.db.execute(text(SET_EF_SEARCH_SQL), {"ef_search": ef_search})

        # Cosine distance: pgvector's <=> operator. Similarity = 1 - distance.
        distance_expr = SemanticMemoryEntry.embedding.cosine_distance(query_embedding)

        stmt = (
            select(SemanticMemoryEntry, distance_expr.label("distance"))
            .order_by(distance_expr)
            .limit(top_k * 2)   # Over-fetch before threshold filtering
        )
        if domain:
            stmt = stmt.where(SemanticMemoryEntry.domain == domain)
        if exclude_contested:
            stmt = stmt.where(SemanticMemoryEntry.is_contested.is_(False))

        result = await self.db.execute(stmt)
        rows = result.all()

        results = []
        for entry, distance in rows:
            similarity = 1.0 - float(distance)
            if similarity < similarity_threshold:
                continue

            effective_confidence = _decay_confidence(
                entry.confidence, entry.last_reinforced_at
            )
            entry_dict = entry.to_dict()
            entry_dict["effective_confidence"] = round(effective_confidence, 4)

            results.append(SemanticSearchResult(entry=entry_dict, similarity=round(similarity, 4)))

            if len(results) >= top_k:
                break

        log.info("semantic_memory.search",
                 query=query[:60],
                 n_results=len(results),
                 threshold=similarity_threshold)
        return results

    async def format_for_agent_context(
        self, query: str, top_k: int = 5, domain: str | None = None
    ) -> str:
        """
        The primary integration point: called by the Planner agent before
        research begins. Returns a ready-to-inject context block, or an
        empty string if nothing relevant exists — callers don't need to
        branch on "did we find anything."
        """
        results = await self.search(query, top_k=top_k, domain=domain)
        if not results:
            return ""

        lines = ["PRIOR KNOWLEDGE FROM PAST RESEARCH (verify if load-bearing to your conclusion):"]
        for r in results:
            entry = r.entry
            flag = " [CONTESTED]" if entry["is_contested"] else ""
            lines.append(
                f"- [{entry['effective_confidence']:.0%} confidence, "
                f"corroborated {entry['corroboration_count']}x{flag}] {entry['content']}"
            )
        return "\n".join(lines)

    # ── Consolidation (episodic → semantic) ─────────────────────────────────

    async def consolidate(
        self, candidate: ConsolidationCandidate
    ) -> tuple[SemanticMemoryEntry, str]:
        """
        Attempt to fold one episode's conclusion into semantic memory.

        Returns (entry, action) where action is one of:
          "created"       — no similar entry existed, new fact recorded
          "reinforced"     — matched an existing entry, corroboration incremented
          "conflict"       — matched an existing entry but claim CONTRADICTS it

        This is called by a background consolidation worker after each
        session completes, NOT inline in the request path — consolidation
        involves an LLM call to check for semantic contradiction, which is
        too slow for the critical path.
        """
        content_hash = _content_hash(candidate.content)

        exact_match_stmt = select(SemanticMemoryEntry).where(
            SemanticMemoryEntry.content_hash == content_hash
        )
        exact_match = (await self.db.execute(exact_match_stmt)).scalar_one_or_none()

        if exact_match:
            return await self._reinforce(exact_match, candidate.episode_id), "reinforced"

        similar = await self.search(
            candidate.content, top_k=1, similarity_threshold=0.85
        )

        if not similar:
            entry = await self._create_entry(candidate)
            return entry, "created"

        existing_entry_id = UUID(similar[0].entry["id"])
        stmt = select(SemanticMemoryEntry).where(SemanticMemoryEntry.id == existing_entry_id)
        existing = (await self.db.execute(stmt)).scalar_one()

        is_contradiction = await self._check_contradiction(
            existing.content, candidate.content
        )

        if is_contradiction:
            await self._flag_conflict(existing, candidate)
            return existing, "conflict"
        else:
            return await self._reinforce(existing, candidate.episode_id), "reinforced"

    async def _create_entry(self, candidate: ConsolidationCandidate) -> SemanticMemoryEntry:
        entry = SemanticMemoryEntry(
            content=candidate.content,
            content_hash=_content_hash(candidate.content),
            embedding=candidate.embedding,
            topics=candidate.topics,
            domain=candidate.domain,
            confidence=candidate.confidence,
            corroboration_count=1,
            source_episode_ids=[candidate.episode_id],
        )
        self.db.add(entry)
        await self.db.commit()
        log.info("semantic_memory.entry_created",
                 content=candidate.content[:80],
                 domain=candidate.domain)
        return entry

    async def _reinforce(
        self, entry: SemanticMemoryEntry, episode_id: UUID
    ) -> SemanticMemoryEntry:
        """
        Strengthen an existing fact: increment corroboration, boost
        confidence (diminishing returns via sqrt-like scaling so the 10th
        corroboration matters less than the 2nd), refresh reinforcement
        timestamp to reset the decay clock.
        """
        entry.corroboration_count += 1
        entry.reinforcement_count += 1
        entry.last_reinforced_at = datetime.now(timezone.utc)

        if episode_id not in (entry.source_episode_ids or []):
            entry.source_episode_ids = list(entry.source_episode_ids or []) + [episode_id]

        # Diminishing-returns confidence boost — asymptotic approach to 0.97
        boost = 0.06 / (entry.corroboration_count ** 0.5)
        entry.confidence = min(entry.confidence + boost, 0.97)

        await self.db.commit()
        log.info("semantic_memory.reinforced",
                 entry_id=str(entry.id),
                 new_confidence=entry.confidence,
                 corroboration=entry.corroboration_count)
        return entry

    async def _check_contradiction(self, existing_claim: str, new_claim: str) -> bool:
        """
        LLM-based contradiction check between two semantically similar
        claims. This is intentionally conservative — high similarity alone
        doesn't mean contradiction (two claims can be near-duplicates in
        embedding space while being fully consistent, e.g. paraphrases).
        """
        from app.core.llm_client import LLMClient, TokenBudget
        import json as _json

        llm = LLMClient("critic", TokenBudget(max_tokens=5000))
        system = "Determine if two claims genuinely contradict each other. Return ONLY valid JSON."
        prompt = f"""Claim A (existing): {existing_claim}
Claim B (new): {new_claim}

Do these claims genuinely CONTRADICT each other (not just discuss different aspects)?
Return JSON: {{"contradicts": true/false, "reason": "one sentence"}}"""

        try:
            response = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=128,
                temperature=0.0,
            )
            data = _json.loads(response.strip())
            return bool(data.get("contradicts", False))
        except Exception as exc:
            log.warning("semantic_memory.contradiction_check_failed", error=str(exc))
            return False   # Fail safe: don't block consolidation on LLM errors

    async def _flag_conflict(
        self, existing: SemanticMemoryEntry, candidate: ConsolidationCandidate
    ) -> None:
        """
        Record the conflict for Critic review, and mark the existing entry
        as contested. Confidence is NOT overwritten — both claims persist
        until a human or the Critic agent explicitly resolves the conflict.
        """
        conflict = SemanticConflict(
            semantic_entry_id=existing.id,
            conflicting_episode_id=candidate.episode_id,
            conflicting_claim=candidate.content,
            existing_claim=existing.content,
        )
        self.db.add(conflict)

        existing.is_contested = True
        existing.contested_reason = f"Conflicts with newer claim: {candidate.content[:200]}"
        existing.contradiction_count += 1

        await self.db.commit()
        log.warning("semantic_memory.conflict_flagged",
                   entry_id=str(existing.id),
                   existing_claim=existing.content[:80],
                   new_claim=candidate.content[:80])

    # ── Conflict resolution (called by Critic agent) ────────────────────────

    async def resolve_conflict(
        self, conflict_id: UUID, resolution: str, winning_claim: str | None = None
    ) -> None:
        stmt = select(SemanticConflict).where(SemanticConflict.id == conflict_id)
        conflict = (await self.db.execute(stmt)).scalar_one_or_none()
        if not conflict:
            raise SemanticMemoryError(f"Conflict {conflict_id} not found")

        conflict.resolved = True
        conflict.resolution = resolution

        if winning_claim:
            entry_stmt = select(SemanticMemoryEntry).where(
                SemanticMemoryEntry.id == conflict.semantic_entry_id
            )
            entry = (await self.db.execute(entry_stmt)).scalar_one()
            entry.content = winning_claim
            entry.is_contested = False
            entry.contested_reason = None

        await self.db.commit()

    async def get_pending_conflicts(self, limit: int = 20) -> list[dict[str, Any]]:
        stmt = (
            select(SemanticConflict)
            .where(SemanticConflict.resolved.is_(False))
            .order_by(SemanticConflict.created_at.desc())
            .limit(limit)
        )
        result = await self.db.execute(stmt)
        return [
            {
                "id": str(c.id),
                "existing_claim": c.existing_claim,
                "conflicting_claim": c.conflicting_claim,
                "created_at": c.created_at.isoformat(),
            }
            for c in result.scalars().all()
        ]

    # ── Maintenance ──────────────────────────────────────────────────────────

    async def get_stats(self) -> dict[str, Any]:
        stmt = select(
            func.count(SemanticMemoryEntry.id),
            func.avg(SemanticMemoryEntry.confidence),
            func.avg(SemanticMemoryEntry.corroboration_count),
            func.count(SemanticMemoryEntry.id).filter(SemanticMemoryEntry.is_contested.is_(True)),
        )
        result = await self.db.execute(stmt)
        row = result.one()
        return {
            "total_entries": row[0] or 0,
            "avg_confidence": round(float(row[1]), 4) if row[1] else None,
            "avg_corroboration": round(float(row[2]), 2) if row[2] else None,
            "contested_entries": row[3] or 0,
        }
