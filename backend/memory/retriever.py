"""
╔══════════════════════════════════════════════════════════════════════╗
║  MEMORY RETRIEVER — Unified orchestration layer                      ║
║                                                                      ║
║  This is the ONLY module other parts of MARS should import from.     ║
║  Agents never touch WorkingMemory, EpisodicMemoryStore, or            ║
║  SemanticMemoryStore directly — they call MemoryRetriever, which       ║
║  coordinates across tiers and hides the plumbing.                     ║
║                                                                      ║
║  Two operations dominate:                                            ║
║                                                                      ║
║  1. retrieve_context(query) — called by the Planner BEFORE research  ║
║     begins. Checks episodic memory's fast-path topic index first      ║
║     (cheap), then falls back to semantic memory's vector search       ║
║     (more expensive, catches paraphrases/related-but-not-identical    ║
║     topics). Returns a single formatted context block.                ║
║                                                                      ║
║  2. consolidate_session(session_id) — called by a background worker   ║
║     AFTER a pipeline completes. Reads the episode from episodic        ║
║     memory, extracts durable conclusions via LLM, embeds each one,     ║
║     and folds them into semantic memory via SemanticMemoryStore.       ║
║     consolidate_session。                                              ║
║                                                                      ║
║  The retrieval-before / consolidation-after split is deliberate:      ║
║  retrieval must be fast (blocks the critical path), consolidation      ║
║  can be slow (runs async, off the critical path).                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.llm_client import LLMClient, TokenBudget
from app.memory.working import WorkingMemory, create_working_memory
from app.memory.episodic import (
    EpisodicMemoryStore,
    EpisodeWriteRequest,
    EpisodicMemoryError,
)
from app.memory.semantic import (
    SemanticMemoryStore,
    ConsolidationCandidate,
    SemanticMemoryError,
)

log = structlog.get_logger(__name__)

MAX_CONSOLIDATION_FACTS_PER_SESSION = 5   # Cap extracted facts per episode
MIN_CONFIDENCE_FOR_CONSOLIDATION = 0.55   # Don't consolidate low-confidence noise


# ── Data transfer objects ──────────────────────────────────────────────────────

@dataclass
class RetrievedContext:
    """
    Combined context from episodic + semantic memory, ready to inject
    into an agent's prompt. Callers get one object regardless of which
    tier(s) actually had relevant data.
    """
    formatted_context: str                  # Ready-to-inject text block
    episodic_hits: int = 0
    semantic_hits: int = 0
    has_content: bool = False
    source_episode_ids: list[str] = field(default_factory=list)
    source_semantic_ids: list[str] = field(default_factory=list)
    retrieval_latency_ms: float | None = None


@dataclass
class ConsolidationResult:
    session_id: str
    facts_extracted: int = 0
    facts_created: int = 0
    facts_reinforced: int = 0
    facts_conflicted: int = 0
    errors: list[str] = field(default_factory=list)


class MemoryRetrieverError(Exception):
    pass


class MemoryRetriever:
    """
    The single entry point for memory operations across the pipeline.
    Instantiated once per request/session — holds references to the
    per-tier stores but adds no state of its own.
    """

    def __init__(
        self,
        db_session: AsyncSession,
        episodic: EpisodicMemoryStore | None = None,
        semantic: SemanticMemoryStore | None = None,
    ):
        self.db = db_session
        self.episodic = episodic or EpisodicMemoryStore(db_session)
        self.semantic = semantic or SemanticMemoryStore(db_session)

    # ── Retrieval (pre-research, on the critical path — must be fast) ──────

    async def retrieve_context(
        self,
        query: str,
        user_id: UUID | None = None,
        domain: str | None = None,
        max_episodic: int = 3,
        max_semantic: int = 5,
    ) -> RetrievedContext:
        """
        The main retrieval entry point. Called once by the Planner before
        the pipeline's Researcher agents begin work.

        Strategy:
          1. Fast-path: exact/fuzzy topic match in episodic memory index.
             Cheap SQL query, catches "I researched this exact thing before."
          2. Semantic search in tier 3 for related-but-not-identical topics.
             Catches paraphrases and conceptually adjacent prior research.
          3. Merge and format as a single context block.

        Both lookups run even if the first returns hits — episodic gives
        specific past events ("I researched X on March 3rd and concluded Y"),
        semantic gives distilled cross-session facts. They're complementary,
        not redundant.
        """
        start = datetime.now(timezone.utc)

        episodic_result = await self.episodic.find_by_topic(
            topic=query, user_id=user_id, limit=max_episodic, fuzzy=True
        )

        semantic_results = await self.semantic.search(
            query=query, top_k=max_semantic, domain=domain
        )

        context_parts: list[str] = []
        source_episode_ids: list[str] = []
        source_semantic_ids: list[str] = []

        if episodic_result.episodes:
            context_parts.append("RELATED PAST RESEARCH SESSIONS:")
            for ep in episodic_result.episodes:
                context_parts.append(
                    f"- On {ep['created_at'][:10]}, researched \"{ep['query'][:100]}\" "
                    f"and settled at {ep['final_confidence']:.0%} confidence "
                    f"({ep['contradictions_found']} contradiction(s) found, "
                    f"{ep['contradictions_resolved']} resolved)."
                    if ep.get("final_confidence") is not None
                    else f"- On {ep['created_at'][:10]}, researched \"{ep['query'][:100]}\"."
                )
                source_episode_ids.append(ep["id"])

        if semantic_results:
            if context_parts:
                context_parts.append("")   # Blank line separator
            context_parts.append("DISTILLED PRIOR KNOWLEDGE (cross-session facts):")
            for r in semantic_results:
                entry = r.entry
                flag = " ⚠ CONTESTED" if entry["is_contested"] else ""
                context_parts.append(
                    f"- [{entry['effective_confidence']:.0%} confidence, "
                    f"corroborated {entry['corroboration_count']}x{flag}] {entry['content']}"
                )
                source_semantic_ids.append(entry["id"])

        formatted = "\n".join(context_parts)
        has_content = bool(episodic_result.episodes or semantic_results)

        latency_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        log.info("memory_retriever.context_retrieved",
                 query=query[:60],
                 episodic_hits=len(episodic_result.episodes),
                 semantic_hits=len(semantic_results),
                 latency_ms=round(latency_ms, 1))

        return RetrievedContext(
            formatted_context=formatted,
            episodic_hits=len(episodic_result.episodes),
            semantic_hits=len(semantic_results),
            has_content=has_content,
            source_episode_ids=source_episode_ids,
            source_semantic_ids=source_semantic_ids,
            retrieval_latency_ms=round(latency_ms, 1),
        )

    # ── Session finalization (write to episodic — end of pipeline run) ─────

    async def finalize_session(
        self,
        session_id: UUID,
        query: str,
        user_id: UUID | None,
        domain: str,
        status: str,
        final_confidence: float | None,
        topics: list[str],
        settled_beliefs: dict[str, Any],
        contradictions_found: int,
        contradictions_resolved: int,
        source_urls: list[str],
        avg_source_trust: float | None,
        total_tokens: int,
        duration_seconds: float,
        refinement_iterations: int,
        report_summary: str | None,
        full_report_json: dict[str, Any] | None,
        error_message: str | None = None,
    ) -> UUID:
        """
        Write the completed session to episodic memory (tier 2), and
        delete its working memory (tier 1) — the transient scratchpad is
        no longer needed once the durable record exists.

        Does NOT trigger semantic consolidation inline — that's a
        separate async step (see consolidate_session) so a slow LLM-based
        fact-extraction pass never delays the user-facing pipeline
        completion event.
        """
        request = EpisodeWriteRequest(
            session_id=session_id,
            query=query,
            user_id=user_id,
            query_domain=domain,
            status=status,
            final_confidence=final_confidence,
            topics=topics,
            settled_beliefs=settled_beliefs,
            contradictions_found=contradictions_found,
            contradictions_resolved=contradictions_resolved,
            source_urls=source_urls,
            avg_source_trust=avg_source_trust,
            total_tokens=total_tokens,
            duration_seconds=duration_seconds,
            refinement_iterations=refinement_iterations,
            report_summary=report_summary,
            full_report_json=full_report_json,
            error_message=error_message,
        )

        try:
            episode = await self.episodic.record_episode(request)
        except EpisodicMemoryError as exc:
            log.error("memory_retriever.finalize_failed",
                     session_id=str(session_id), error=str(exc))
            raise MemoryRetrieverError(f"Failed to finalize session: {exc}") from exc

        try:
            wm = await create_working_memory(str(session_id))
            await wm.delete()
        except Exception as exc:
            # Working memory cleanup failure is non-fatal — TTL will
            # eventually expire it. Log and move on.
            log.warning("memory_retriever.working_memory_cleanup_failed",
                       session_id=str(session_id), error=str(exc))

        log.info("memory_retriever.session_finalized",
                 session_id=str(session_id),
                 episode_id=str(episode.id))
        return episode.id

    # ── Consolidation (episodic → semantic, async background job) ──────────

    async def consolidate_session(self, session_id: UUID) -> ConsolidationResult:
        """
        Extract durable, cross-session-worthy facts from a completed episode
        and fold them into semantic memory. Called by a background worker
        (Celery task or asyncio.create_task fired off after pipeline
        completion) — never inline in the request path.
        """
        episode = await self.episodic.get_by_session_id(session_id)
        if episode is None:
            return ConsolidationResult(
                session_id=str(session_id),
                errors=[f"Episode not found for session {session_id}"],
            )

        if episode.status != "complete" or episode.final_confidence is None:
            log.info("memory_retriever.skip_consolidation_incomplete",
                     session_id=str(session_id), status=episode.status)
            return ConsolidationResult(session_id=str(session_id))

        if episode.final_confidence < MIN_CONFIDENCE_FOR_CONSOLIDATION:
            log.info("memory_retriever.skip_consolidation_low_confidence",
                     session_id=str(session_id),
                     confidence=episode.final_confidence)
            return ConsolidationResult(session_id=str(session_id))

        result = ConsolidationResult(session_id=str(session_id))

        try:
            facts = await self._extract_durable_facts(episode)
            result.facts_extracted = len(facts)
        except Exception as exc:
            result.errors.append(f"Fact extraction failed: {exc}")
            log.error("memory_retriever.extraction_failed",
                     session_id=str(session_id), error=str(exc))
            return result

        for fact_content, fact_topics in facts:
            try:
                embedding = self.semantic.embed(fact_content)
                candidate = ConsolidationCandidate(
                    episode_id=episode.id,
                    content=fact_content,
                    embedding=embedding,
                    topics=fact_topics,
                    domain=episode.query_domain or "general",
                    confidence=episode.final_confidence,
                )
                _, action = await self.semantic.consolidate(candidate)

                if action == "created":
                    result.facts_created += 1
                elif action == "reinforced":
                    result.facts_reinforced += 1
                elif action == "conflict":
                    result.facts_conflicted += 1

            except SemanticMemoryError as exc:
                result.errors.append(f"Consolidation failed for '{fact_content[:50]}': {exc}")
                log.warning("memory_retriever.consolidation_entry_failed",
                          session_id=str(session_id), error=str(exc))

        log.info("memory_retriever.consolidation_complete",
                 session_id=str(session_id),
                 created=result.facts_created,
                 reinforced=result.facts_reinforced,
                 conflicted=result.facts_conflicted,
                 errors=len(result.errors))
        return result

    async def _extract_durable_facts(
        self, episode: Any
    ) -> list[tuple[str, list[str]]]:
        """
        LLM-based extraction of durable, reusable conclusions from a
        completed episode. Returns [(fact_text, topics), ...].

        A "durable" fact is specific, verifiable, and likely to remain
        true for months — not a process description or session-specific
        detail. This mirrors the distillation step in episodic-to-semantic
        consolidation described in multi-agent memory literature: the team
        reflects on a completed project and extracts what should persist.
        """
        llm = LLMClient("researcher", TokenBudget(max_tokens=10000))

        beliefs_summary = json.dumps(episode.settled_beliefs or {}, indent=None)[:2000]
        report_excerpt = (episode.report_summary or "")[:1500]

        system = (
            "Extract durable knowledge conclusions from a research session. "
            "Return ONLY valid JSON."
        )
        prompt = f"""Original query: {episode.query}

Settled beliefs: {beliefs_summary}

Report summary: {report_excerpt}

Extract up to {MAX_CONSOLIDATION_FACTS_PER_SESSION} durable, reusable factual
conclusions. A durable fact is:
- Specific and verifiable (not a vague generality)
- Likely to remain true for 6+ months
- Useful context for a FUTURE, different research session on a related topic

Return JSON array:
[
  {{"content": "Specific factual conclusion in one sentence", "topics": ["topic1", "topic2"]}}
]

If nothing in this session is durable enough to be worth remembering long-term,
return an empty array []."""

        try:
            response = await llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=1024,
                temperature=0.0,
            )
            raw = json.loads(response.strip())
            if not isinstance(raw, list):
                return []

            facts = []
            for item in raw[:MAX_CONSOLIDATION_FACTS_PER_SESSION]:
                if isinstance(item, dict) and item.get("content"):
                    facts.append((item["content"], item.get("topics", [])))
            return facts

        except Exception as exc:
            log.warning("memory_retriever.fact_extraction_parse_failed", error=str(exc))
            return []

    # ── Dashboard / observability ────────────────────────────────────────────

    async def get_memory_health(self) -> dict[str, Any]:
        """Combined stats across tiers 2 and 3, for the ops dashboard."""
        episodic_stats = await self.episodic.get_stats()
        semantic_stats = await self.semantic.get_stats()
        pending_conflicts = await self.semantic.get_pending_conflicts(limit=5)

        return {
            "episodic": episodic_stats,
            "semantic": semantic_stats,
            "pending_conflicts": pending_conflicts,
            "pending_conflict_count": len(pending_conflicts),
        }
