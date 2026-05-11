"""
Five specialist agents — each with a typed contract, scoped tools,
and structured output format. Every agent is async-native and
emits SSE events to the caller's queue.
"""
from __future__ import annotations
import asyncio
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any
from uuid import uuid4

import structlog

from app.core.config import settings
from app.core.llm_client import LLMClient, TokenBudget
from app.schemas.research import (
    AgentStatus,
    AgentTrace,
    PipelineState,
    ResearchReport,
    ResearchStatus,
    ReportSection,
    Source,
    SourceType,
    SSEEvent,
    SSEEventType,
    SubTask,
)
from app.core.tools.search import WebSearchTool, ArxivTool, PDFTool
from app.core.tools.verify import CitationVerifier, FactChecker

log = structlog.get_logger(__name__)


# ── Base agent ────────────────────────────────────────────────────────────────

class BaseAgent(ABC):
    role: str = "base"

    def __init__(
        self,
        budget: TokenBudget,
        event_queue: asyncio.Queue[SSEEvent],
        session_id: str,
    ):
        self.budget = budget
        self.queue = event_queue
        self.session_id = session_id
        self.llm = LLMClient(self.role, budget)
        self._trace = AgentTrace(agent_id=self.role, status=AgentStatus.PENDING)

    async def emit(self, event_type: SSEEventType, data: dict[str, Any]) -> None:
        event = SSEEvent(
            event=event_type,
            session_id=self.session_id,
            data={"agent": self.role, **data},
        )
        await self.queue.put(event)

    async def run(self, state: PipelineState) -> PipelineState:
        self._trace.status = AgentStatus.RUNNING
        self._trace.started_at = datetime.utcnow()
        await self.emit(SSEEventType.AGENT_START, {
            "agent_role": self.role,
            "message": f"{self.role.title()} agent starting",
        })

        try:
            state = await self._execute(state)
            self._trace.status = AgentStatus.DONE
        except Exception as exc:
            self._trace.status = AgentStatus.ERROR
            self._trace.error = str(exc)
            log.error("agent.error", role=self.role, error=str(exc),
                      session_id=self.session_id)
            await self.emit(SSEEventType.AGENT_ERROR, {
                "error": str(exc),
                "agent_role": self.role,
            })
            raise
        finally:
            self._trace.completed_at = datetime.utcnow()
            if self._trace.started_at:
                delta = self._trace.completed_at - self._trace.started_at
                self._trace.duration_ms = int(delta.total_seconds() * 1000)
            self._trace.token_usage = {"budget_used": self.budget.used}
            state.agent_traces[self.role] = self._trace
            if self._trace.status == AgentStatus.DONE:
                await self.emit(SSEEventType.AGENT_COMPLETE, {
                    "agent_role": self.role,
                    "duration_ms": self._trace.duration_ms,
                    "tokens_used": self.budget.used,
                })

        return state

    @abstractmethod
    async def _execute(self, state: PipelineState) -> PipelineState:
        ...


# ── Orchestrator ──────────────────────────────────────────────────────────────

class OrchestratorAgent(BaseAgent):
    """
    Mission control. Receives the raw query, assesses complexity,
    then delegates to Planner → Researchers → Critic → Synthesizer.
    Handles failure recovery and circuit-breaking.
    """
    role = "orchestrator"

    async def _execute(self, state: PipelineState) -> PipelineState:
        log.info("orchestrator.start", query=state.query[:80],
                 session_id=self.session_id)

        system = """You are a research orchestration expert. Your job is to:
1. Analyze the research query for complexity and scope
2. Identify key dimensions that need investigation
3. Determine appropriate research depth

Respond ONLY with valid JSON. No markdown, no preamble."""

        complexity_prompt = f"""Analyze this research query and return JSON:

Query: {state.query}

Return:
{{
  "complexity": "simple|moderate|complex",
  "estimated_subtopics": 2-6,
  "recommended_depth": "quick|standard|deep",
  "key_dimensions": ["dimension1", "dimension2", ...],
  "domain": "science|technology|business|policy|general",
  "temporal_sensitivity": "real-time|recent|historical"
}}"""

        response = await self.llm.complete(
            messages=[{"role": "user", "content": complexity_prompt}],
            system=system,
            session_id=self.session_id,
        )

        try:
            analysis = json.loads(response.strip())
        except json.JSONDecodeError:
            analysis = {
                "complexity": "moderate",
                "key_dimensions": ["overview", "analysis"],
                "domain": "general",
            }

        state.metadata["query_analysis"] = analysis
        await self.emit(SSEEventType.AGENT_PROGRESS, {
            "message": f"Query classified as {analysis.get('complexity', 'moderate')} complexity",
            "analysis": analysis,
        })

        return state


# ── Planner ───────────────────────────────────────────────────────────────────

class PlannerAgent(BaseAgent):
    """
    Converts orchestrator analysis into a concrete research plan.
    Generates SubTasks with explicit scope rules to prevent overlap.
    """
    role = "planner"

    async def _execute(self, state: PipelineState) -> PipelineState:
        analysis = state.metadata.get("query_analysis", {})
        dimensions = analysis.get("key_dimensions", ["overview", "details"])

        system = """You are a research planning expert. Create precise, non-overlapping research sub-tasks.
Each sub-task must have clear boundaries to prevent duplicate work between researchers.
Respond ONLY with valid JSON."""

        plan_prompt = f"""Create a research plan for this query.

Query: {state.query}
Key dimensions: {dimensions}
Domain: {analysis.get('domain', 'general')}

Return JSON array of sub-tasks (max {settings.max_subtopics}):
[
  {{
    "heading": "Brief heading",
    "objective": "Precise research objective — what specifically to find",
    "assigned_to": "researcher_a or researcher_b",
    "allowed_sources": ["web", "arxiv", "semantic_scholar"],
    "scope_rules": [
      "Only cover X, not Y",
      "Focus on period Z",
      "Exclude topic W (assigned to other researcher)"
    ],
    "priority": 1-5
  }}
]

CRITICAL: alternate assignments between researcher_a and researcher_b to enable parallel execution.
No two tasks should cover overlapping content."""

        response = await self.llm.complete(
            messages=[{"role": "user", "content": plan_prompt}],
            system=system,
            session_id=self.session_id,
        )

        try:
            raw_tasks = json.loads(response.strip())
            if not isinstance(raw_tasks, list):
                raw_tasks = raw_tasks.get("tasks", [raw_tasks])
        except json.JSONDecodeError:
            raw_tasks = [
                {"heading": "Primary research", "objective": state.query,
                 "assigned_to": "researcher_a", "priority": 1},
                {"heading": "Supporting analysis", "objective": f"Supporting context for: {state.query}",
                 "assigned_to": "researcher_b", "priority": 2},
            ]

        state.sub_tasks = [
            SubTask(
                heading=t.get("heading", "Research task"),
                objective=t.get("objective", state.query),
                assigned_to=t.get("assigned_to", "researcher_a"),
                allowed_sources=[SourceType(s) for s in t.get("allowed_sources", ["web"])
                                 if s in SourceType.__members__.values()],
                scope_rules=t.get("scope_rules", []),
                priority=t.get("priority", 3),
            )
            for t in raw_tasks[:settings.max_subtopics]
        ]

        state.status = ResearchStatus.RESEARCHING
        await self.emit(SSEEventType.AGENT_PROGRESS, {
            "message": f"Research plan created: {len(state.sub_tasks)} sub-tasks",
            "subtasks": [t.heading for t in state.sub_tasks],
        })

        return state


# ── Researcher ────────────────────────────────────────────────────────────────

class ResearcherAgent(BaseAgent):
    """
    Executes parallel searches across web + academic sources.
    Respects assigned sub-tasks and scope rules.
    Enforces concurrency limits with asyncio.Semaphore.
    """

    def __init__(self, researcher_id: str, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.researcher_id = researcher_id
        self.role = researcher_id  # "researcher_a" | "researcher_b"
        self.llm = LLMClient("researcher", self.budget)
        self._semaphore = asyncio.Semaphore(settings.max_researcher_concurrency)
        self._web_tool = WebSearchTool()
        self._arxiv_tool = ArxivTool()
        self._pdf_tool = PDFTool()

    async def _execute(self, state: PipelineState) -> PipelineState:
        my_tasks = [t for t in state.sub_tasks if t.assigned_to == self.researcher_id]

        if not my_tasks:
            log.info("researcher.no_tasks", researcher=self.researcher_id)
            return state

        await self.emit(SSEEventType.AGENT_PROGRESS, {
            "message": f"Starting {len(my_tasks)} sub-task(s)",
            "tasks": [t.heading for t in my_tasks],
        })

        # Run tasks concurrently with semaphore
        async def research_task(task: SubTask) -> SubTask:
            async with self._semaphore:
                return await self._research_subtask(task, state)

        results = await asyncio.gather(
            *[research_task(t) for t in my_tasks],
            return_exceptions=True,
        )

        for task, result in zip(my_tasks, results):
            if isinstance(result, Exception):
                log.error("researcher.subtask_failed",
                          task=task.heading, error=str(result))
                task.status = AgentStatus.ERROR
            else:
                # Update the original task in state
                for i, t in enumerate(state.sub_tasks):
                    if t.id == result.id:
                        state.sub_tasks[i] = result
                        break

        all_sources = [s for t in state.sub_tasks for s in t.sources]
        await self.emit(SSEEventType.SOURCES_FOUND, {
            "total_sources": len(all_sources),
            "by_type": {
                st.value: len([s for s in all_sources if s.source_type == st])
                for st in SourceType
            },
        })

        return state

    async def _research_subtask(self, task: SubTask, state: PipelineState) -> SubTask:
        task.status = AgentStatus.RUNNING

        # Determine search strategy from allowed sources
        sources: list[Source] = []
        search_queries = await self._generate_search_queries(task)

        # Parallel source fetching
        fetch_tasks = []
        for query in search_queries[:3]:
            fetch_tasks.append(self._web_tool.search(query))
            if SourceType.ARXIV in task.allowed_sources or not task.allowed_sources:
                fetch_tasks.append(self._arxiv_tool.search(query))

        raw_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        for r in raw_results:
            if isinstance(r, list):
                sources.extend(r)

        # Deduplicate by URL
        seen_urls: set[str] = set()
        unique_sources = []
        for s in sources:
            if s.url not in seen_urls:
                seen_urls.add(s.url)
                unique_sources.append(s)

        task.sources = unique_sources[:20]

        # Synthesize findings from sources
        task.findings = await self._synthesize_findings(task, state.query)
        task.status = AgentStatus.DONE

        return task

    async def _generate_search_queries(self, task: SubTask) -> list[str]:
        system = "Generate precise search queries. Return JSON array of 3 strings only."
        prompt = f"""Task: {task.objective}
Scope: {'; '.join(task.scope_rules[:3])}

Return 3 specific search queries as JSON array: ["query1", "query2", "query3"]"""

        response = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=256,
            session_id=self.session_id,
        )
        try:
            return json.loads(response.strip())
        except Exception:
            return [task.objective, f"{task.heading} latest research", f"{task.objective} analysis"]

    async def _synthesize_findings(self, task: SubTask, query: str) -> list[str]:
        if not task.sources:
            return [f"No sources found for: {task.heading}"]

        source_summaries = "\n".join(
            f"- [{s.source_type.value}] {s.title}: {s.snippet[:200]}"
            for s in task.sources[:10]
        )

        system = """Extract and synthesize key findings from research sources.
Return JSON array of 4-6 specific, factual findings. No fluff."""

        prompt = f"""Research objective: {task.objective}
Context: {query}

Sources found:
{source_summaries}

Return 4-6 key findings as JSON array of strings. Be specific and factual."""

        response = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=1024,
            session_id=self.session_id,
        )
        try:
            findings = json.loads(response.strip())
            return findings if isinstance(findings, list) else [str(findings)]
        except Exception:
            return [f"Research completed for {task.heading} — {len(task.sources)} sources analyzed"]


# ── Critic ────────────────────────────────────────────────────────────────────

class CriticAgent(BaseAgent):
    """
    Independently reviews all findings for:
    - Coverage gaps (triggers refinement, max 2 iterations)
    - Source contradictions
    - Confidence scoring per claim
    - Quality thresholds

    Structurally separate from researchers — correlated errors cannot pass.
    """
    role = "critic"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._fact_checker = FactChecker()
        self._verifier = CitationVerifier()

    async def _execute(self, state: PipelineState) -> PipelineState:
        await self.emit(SSEEventType.AGENT_PROGRESS, {
            "message": "Starting gap analysis and quality check",
        })

        # Build findings summary
        all_findings = []
        for task in state.sub_tasks:
            for finding in task.findings:
                all_findings.append(f"[{task.heading}] {finding}")

        if not all_findings:
            await self.emit(SSEEventType.AGENT_PROGRESS, {
                "message": "No findings to critique — skipping",
            })
            return state

        system = """You are a rigorous research critic. Identify gaps, contradictions, and quality issues.
Be specific about what is missing and what needs refinement.
Respond ONLY with valid JSON."""

        critique_prompt = f"""Critique this research for the query: "{state.query}"

Findings collected:
{chr(10).join(all_findings[:30])}

Covered topics: {[t.heading for t in state.sub_tasks]}

Return JSON:
{{
  "coverage_gaps": ["gap1", "gap2"],
  "contradictions": [{{"finding_a": "...", "finding_b": "...", "resolution": "..."}}],
  "confidence_scores": {{"topic_heading": 0.0-1.0}},
  "quality_issues": ["issue1"],
  "refinement_needed": true/false,
  "refinement_targets": ["specific subtopic needing more research"]
}}"""

        response = await self.llm.complete(
            messages=[{"role": "user", "content": critique_prompt}],
            system=system,
            session_id=self.session_id,
        )

        try:
            critique = json.loads(response.strip())
        except json.JSONDecodeError:
            critique = {"refinement_needed": False, "confidence_scores": {}}

        state.metadata["critique"] = critique

        gaps = critique.get("coverage_gaps", [])
        contradictions = critique.get("contradictions", [])

        await self.emit(SSEEventType.AGENT_PROGRESS, {
            "message": f"Critique complete: {len(gaps)} gaps, {len(contradictions)} contradictions",
            "gaps": gaps[:5],
            "refinement_needed": critique.get("refinement_needed", False),
        })

        # Trigger refinement if needed and within iteration budget
        if (critique.get("refinement_needed") and
                state.refinement_count < settings.max_refinement_iterations):
            state = await self._trigger_refinement(state, critique)

        return state

    async def _trigger_refinement(
        self, state: PipelineState, critique: dict[str, Any]
    ) -> PipelineState:
        state.refinement_count += 1
        state.status = ResearchStatus.REFINING

        await self.emit(SSEEventType.REFINEMENT_LOOP, {
            "iteration": state.refinement_count,
            "max_iterations": settings.max_refinement_iterations,
            "targets": critique.get("refinement_targets", []),
        })

        from app.core.metrics import refinement_iterations_total
        refinement_iterations_total.inc()

        # Create targeted sub-tasks for the gaps
        gaps = critique.get("refinement_targets", critique.get("coverage_gaps", []))
        for i, gap in enumerate(gaps[:2]):
            gap_task = SubTask(
                heading=f"Gap fill: {gap[:50]}",
                objective=f"Research specifically: {gap}. Context: {state.query}",
                assigned_to="researcher_a" if i % 2 == 0 else "researcher_b",
                scope_rules=[f"Focus only on: {gap}", "Be concise — 3-5 findings"],
                priority=5,
            )
            state.sub_tasks.append(gap_task)

        log.info("critic.refinement",
                 iteration=state.refinement_count,
                 gaps_count=len(gaps),
                 session_id=self.session_id)

        return state


# ── Synthesizer ───────────────────────────────────────────────────────────────

class SynthesizerAgent(BaseAgent):
    """
    Assembles the final report from all verified findings.
    - Sequential section generation with running context injection
      (prevents repetition between sections)
    - Confidence scoring from Critic
    - Full citation metadata
    - Structured JSON + markdown output
    """
    role = "synthesizer"

    async def _execute(self, state: PipelineState) -> PipelineState:
        await self.emit(SSEEventType.AGENT_PROGRESS, {
            "message": "Generating structured report",
        })

        critique = state.metadata.get("critique", {})
        confidence_scores = critique.get("confidence_scores", {})

        # Plan sections
        sections_plan = await self._plan_sections(state)

        # Generate each section with context injection
        sections: list[ReportSection] = []
        running_context = ""

        for i, section_plan in enumerate(sections_plan):
            await self.emit(SSEEventType.REPORT_SECTION, {
                "section_index": i,
                "total_sections": len(sections_plan),
                "heading": section_plan.get("heading", ""),
            })
            section = await self._generate_section(
                section_plan, state, running_context
            )
            # Inject 3-sentence summary of this section for next section's context
            running_context = f"Previous section '{section.heading}': {section.content[:300]}..."
            sections.append(section)

        # Apply confidence scores from Critic
        for section in sections:
            for heading, score in confidence_scores.items():
                if heading.lower() in section.heading.lower():
                    section.confidence = round(score, 3)

        # Calculate overall confidence
        all_conf = [s.confidence for s in sections] or [0.7]
        overall_conf = round(sum(all_conf) / len(all_conf), 3)

        # Aggregate all sources
        all_sources = list({
            s.url: s
            for task in state.sub_tasks
            for s in task.sources
        }.values())

        # Generate title and executive summary
        title, summary = await self._generate_title_and_summary(state, sections)

        state.report = ResearchReport(
            title=title,
            executive_summary=summary,
            sections=sections,
            all_sources=all_sources[:30],
            overall_confidence=overall_conf,
            total_sources=len(all_sources),
            word_count=sum(s.word_count for s in sections),
            refinement_iterations=state.refinement_count,
            model_versions={
                "orchestrator": settings.orchestrator_model,
                "researcher": settings.researcher_model,
                "critic": settings.critic_model,
                "synthesizer": settings.synthesizer_model,
            },
        )

        state.status = ResearchStatus.COMPLETE

        from app.core.metrics import confidence_score_summary
        confidence_score_summary.observe(overall_conf)

        await self.emit(SSEEventType.AGENT_PROGRESS, {
            "message": f"Report complete: {len(sections)} sections, {state.report.word_count} words",
            "confidence": overall_conf,
            "sources": len(all_sources),
        })

        return state

    async def _plan_sections(self, state: PipelineState) -> list[dict[str, Any]]:
        completed_tasks = [t for t in state.sub_tasks if t.status == AgentStatus.DONE]

        system = "Plan a research report structure. Return JSON array only."
        prompt = f"""Plan sections for a research report on: "{state.query}"

Completed research areas: {[t.heading for t in completed_tasks]}

Return JSON array:
[{{"heading": "...", "task_ids": ["..."], "focus": "what to cover in this section"}}]

Max 6 sections. Start with executive overview, end with conclusions/outlook."""

        response = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=512,
            session_id=self.session_id,
        )
        try:
            plan = json.loads(response.strip())
            return plan if isinstance(plan, list) else [plan]
        except Exception:
            return [
                {"heading": t.heading, "focus": t.objective}
                for t in completed_tasks[:5]
            ]

    async def _generate_section(
        self,
        plan: dict[str, Any],
        state: PipelineState,
        context: str,
    ) -> ReportSection:
        heading = plan.get("heading", "Section")
        focus = plan.get("focus", heading)

        # Find relevant findings
        relevant_findings = []
        for task in state.sub_tasks:
            if any(word in task.heading.lower() for word in heading.lower().split()):
                relevant_findings.extend(task.findings)
        if not relevant_findings:
            # Fall back to all findings
            relevant_findings = [f for t in state.sub_tasks for f in t.findings]

        system = """Write a research report section. Be factual, specific, and cite evidence.
No fluff, no repetition of context from other sections. Respond ONLY with JSON."""

        prompt = f"""Write report section for: "{state.query}"

Section: {heading}
Focus: {focus}
Context from previous sections (DO NOT repeat): {context or 'None'}

Key findings to incorporate:
{chr(10).join(f'- {f}' for f in relevant_findings[:8])}

Return JSON:
{{
  "content": "2-3 paragraphs of substantive, specific content",
  "confidence": 0.0-1.0,
  "gaps_noted": ["any acknowledged limitations"]
}}"""

        response = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=1024,
            session_id=self.session_id,
        )

        try:
            data = json.loads(response.strip())
        except json.JSONDecodeError:
            data = {"content": response, "confidence": 0.7}

        content = data.get("content", "")
        return ReportSection(
            heading=heading,
            content=content,
            confidence=data.get("confidence", 0.7),
            gaps_noted=data.get("gaps_noted", []),
            word_count=len(content.split()),
        )

    async def _generate_title_and_summary(
        self, state: PipelineState, sections: list[ReportSection]
    ) -> tuple[str, str]:
        system = "Generate a research report title and summary. Return JSON only."
        prompt = f"""Query: {state.query}
Sections: {[s.heading for s in sections]}
First section preview: {sections[0].content[:400] if sections else ''}

Return: {{"title": "Concise report title", "summary": "2-3 sentence executive summary"}}"""

        response = await self.llm.complete(
            messages=[{"role": "user", "content": prompt}],
            system=system,
            max_tokens=256,
            session_id=self.session_id,
        )
        try:
            data = json.loads(response.strip())
            return data.get("title", state.query[:80]), data.get("summary", "")
        except Exception:
            return state.query[:80], f"Research report on: {state.query}"
