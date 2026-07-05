"""
╔══════════════════════════════════════════════════════════════════════╗
║  MEMORY INTEGRATION — wiring the three tiers into the live pipeline    ║
║                                                                      ║
║  This module contains the four integration points where memory        ║
║  actually changes agent behavior. Everything in working.py,           ║
║  episodic.py, semantic.py, and retriever.py is infrastructure —        ║
║  THIS is where it becomes real.                                       ║
║                                                                      ║
║  Integration point 1 — Planner: retrieves context before planning     ║
║  Integration point 2 — Researcher: writes to working memory live      ║
║  Integration point 3 — Critic: checks semantic memory for known       ║
║                          contradictions before flagging new ones       ║
║  Integration point 4 — Pipeline completion: finalizes episode +        ║
║                          schedules async consolidation                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""
from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

import structlog

from app.memory.retriever import MemoryRetriever, RetrievedContext
from app.memory.working import WorkingMemory

log = structlog.get_logger(__name__)


# ── Integration point 1: Planner ────────────────────────────────────────────

class MemoryAwarePlannerMixin:
    """
    Mix into the existing PlannerAgent class. Adds a memory-retrieval step
    that runs BEFORE the planning LLM call, so retrieved context can be
    injected into the planning prompt.

    Usage in agents.py:
        class PlannerAgent(MemoryAwarePlannerMixin, BaseAgent):
            ...

    This is additive — if memory retrieval fails or finds nothing, planning
    proceeds exactly as before. Memory never blocks the pipeline.
    """

    async def retrieve_memory_context(
        self,
        query: str,
        memory: MemoryRetriever,
        user_id: UUID | None = None,
        domain: str | None = None,
    ) -> RetrievedContext:
        try:
            context = await memory.retrieve_context(
                query=query, user_id=user_id, domain=domain
            )
        except Exception as exc:
            # Memory retrieval failure must never block planning.
            log.warning("memory_integration.retrieval_failed",
                       query=query[:60], error=str(exc))
            return RetrievedContext(formatted_context="", has_content=False)

        if context.has_content:
            log.info("memory_integration.context_injected",
                     episodic_hits=context.episodic_hits,
                     semantic_hits=context.semantic_hits,
                     latency_ms=context.retrieval_latency_ms)

        return context

    def build_planning_prompt_with_memory(
        self,
        base_prompt: str,
        memory_context: RetrievedContext,
    ) -> str:
        """
        Injects retrieved memory context into the planning prompt.
        Explicitly instructs the Planner to treat prior knowledge as
        a STARTING POINT to verify, not a ground truth to assume —
        this prevents memory from becoming a hallucination amplifier
        where old (possibly outdated) conclusions get restated as if
        freshly researched.
        """
        if not memory_context.has_content:
            return base_prompt

        return f"""{memory_context.formatted_context}

IMPORTANT: The above is prior knowledge from past research. Treat it as a
starting hypothesis to VERIFY, not a settled fact — assign research tasks
that specifically re-check anything above that is load-bearing to the
current query, especially anything marked CONTESTED or older than a few
months. Do not simply restate prior conclusions without fresh verification.

---

{base_prompt}"""


# ── Integration point 2: Researcher live write-through ─────────────────────

class MemoryAwareResearcherMixin:
    """
    Mix into ResearcherAgent. After each finding is produced, writes it to
    working memory immediately (not batched at the end of the agent's run) —
    this is what makes the live belief graph and SSE confidence updates
    possible: other parts of the system (the SSE handler, a concurrently
    running Critic) can observe partial progress through working memory's
    pub/sub, rather than waiting for the whole agent to finish.
    """

    async def write_through_working_memory(
        self,
        working_memory: WorkingMemory,
        agent_id: str,
        output_text: str,
        confidence: float,
        token_usage: dict[str, int] | None = None,
    ) -> None:
        try:
            await working_memory.record_agent_output(
                agent_id=agent_id,
                output_text=output_text,
                confidence=confidence,
                token_usage=token_usage,
            )
        except Exception as exc:
            # Working memory write failure is non-fatal for the pipeline —
            # the agent's return value still flows through the normal
            # LangGraph state. We lose live-observability for this step,
            # not correctness.
            log.warning("memory_integration.working_memory_write_failed",
                       agent_id=agent_id, error=str(exc))

    async def update_topic_confidence(
        self,
        working_memory: WorkingMemory,
        topic: str,
        confidence: float,
    ) -> None:
        try:
            await working_memory.update_confidence(topic, confidence)
        except Exception as exc:
            log.warning("memory_integration.confidence_update_failed",
                       topic=topic, error=str(exc))


# ── Integration point 3: Critic cross-checks semantic memory ───────────────

class MemoryAwareCriticMixin:
    """
    Mix into CriticAgent. Before flagging a NEW contradiction, checks
    whether semantic memory already has a contested entry covering the
    same ground — if so, surfaces the EXISTING conflict rather than
    creating a duplicate, and includes prior resolution attempts (if any)
    as context for how to think about it.
    """

    async def check_known_contradictions(
        self,
        memory: MemoryRetriever,
        topic: str,
    ) -> list[dict[str, Any]]:
        try:
            results = await memory.semantic.search(
                query=topic, top_k=3, exclude_contested=False
            )
            contested = [r.entry for r in results if r.entry.get("is_contested")]
            if contested:
                log.info("memory_integration.known_contradiction_found",
                         topic=topic[:60], count=len(contested))
            return contested
        except Exception as exc:
            log.warning("memory_integration.known_contradiction_check_failed",
                       topic=topic[:60], error=str(exc))
            return []


# ── Integration point 4: Pipeline completion → finalize + consolidate ──────

async def finalize_and_consolidate(
    memory: MemoryRetriever,
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
    Called once, at the very end of ResearchPipeline.run(), after the
    Synthesizer produces its report. Two things happen:

      1. finalize_session() — SYNCHRONOUS, blocks pipeline completion.
         This must succeed before the pipeline reports "done" to the user,
         because the episode record IS the durable proof the research
         happened. If this fails, the pipeline should surface an error
         rather than silently losing the episode.

      2. consolidate_session() — ASYNCHRONOUS, fired via asyncio.create_task
         and NOT awaited here. Consolidation involves multiple LLM calls
         (fact extraction + contradiction checking per fact) and can take
         several seconds — the user should see "pipeline complete" the
         moment the report is ready, not wait for memory consolidation.

    This function is the concrete answer to "how does memory actually
    get used, not just built" — it's the wiring, not the infrastructure.
    """
    episode_id = await memory.finalize_session(
        session_id=session_id,
        query=query,
        user_id=user_id,
        domain=domain,
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

    if status == "complete":
        asyncio.create_task(
            _background_consolidate(memory, session_id),
            name=f"consolidate-{session_id}",
        )

    return episode_id


async def _background_consolidate(memory: MemoryRetriever, session_id: UUID) -> None:
    """
    Fire-and-forget wrapper. Exceptions here must not propagate to an
    unhandled-task-exception warning that looks like a pipeline failure —
    they're logged and swallowed, since consolidation is a best-effort
    enrichment step, not a correctness requirement.
    """
    try:
        result = await memory.consolidate_session(session_id)
        log.info("memory_integration.background_consolidation_done",
                 session_id=str(session_id),
                 created=result.facts_created,
                 reinforced=result.facts_reinforced,
                 conflicted=result.facts_conflicted)
    except Exception as exc:
        log.error("memory_integration.background_consolidation_failed",
                 session_id=str(session_id), error=str(exc))
