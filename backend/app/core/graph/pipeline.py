"""
LangGraph graph definition — the production pipeline state machine.

Node flow:
  orchestrator → planner → [researcher_a ∥ researcher_b] → critic
        ↓ (if refinement needed and iterations < max)
      [researcher_a ∥ researcher_b again] → critic (loop)
        ↓ (when done or max iterations reached)
      synthesizer → END

Each node is an async function wrapping the corresponding agent.
State is checkpointed in PostgreSQL via LangGraph's built-in persister.
"""
from __future__ import annotations
import asyncio
from typing import Any, Literal, TypedDict
from uuid import UUID

import structlog
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver  # swap to postgres in prod

from app.core.agents.agents import (
    OrchestratorAgent,
    PlannerAgent,
    ResearcherAgent,
    CriticAgent,
    SynthesizerAgent,
)
from app.core.llm_client import TokenBudget
from app.schemas.research import (
    PipelineState,
    ResearchStatus,
    SSEEvent,
    SSEEventType,
)
from app.core.metrics import pipeline_runs_total, pipeline_duration_seconds, pipeline_active_gauge

log = structlog.get_logger(__name__)


class GraphState(TypedDict):
    """LangGraph state dict — wraps our Pydantic PipelineState."""
    pipeline: dict[str, Any]  # PipelineState.model_dump()
    session_id: str
    iteration: int


def build_pipeline(
    event_queue: asyncio.Queue[SSEEvent],
    session_id: str,
    budget: TokenBudget,
) -> StateGraph:
    """
    Build and compile the research pipeline graph.
    Returns a compiled graph ready for invocation.
    """

    def make_node(AgentClass: type, **kwargs: Any):
        async def node_fn(state: GraphState) -> GraphState:
            pipeline_state = PipelineState(**state["pipeline"])
            agent = AgentClass(
                budget=budget,
                event_queue=event_queue,
                session_id=session_id,
                **kwargs,
            )
            updated_state = await agent.run(pipeline_state)
            return {
                **state,
                "pipeline": updated_state.model_dump(mode="json"),
            }
        return node_fn

    def make_researcher_node(researcher_id: str):
        async def node_fn(state: GraphState) -> GraphState:
            pipeline_state = PipelineState(**state["pipeline"])
            agent = ResearcherAgent(
                researcher_id=researcher_id,
                budget=budget,
                event_queue=event_queue,
                session_id=session_id,
            )
            updated_state = await agent.run(pipeline_state)
            return {
                **state,
                "pipeline": updated_state.model_dump(mode="json"),
            }
        return node_fn

    async def parallel_research(state: GraphState) -> GraphState:
        """Run both researchers concurrently."""
        pipeline_state = PipelineState(**state["pipeline"])

        a = ResearcherAgent("researcher_a", budget, event_queue, session_id)
        b = ResearcherAgent("researcher_b", budget, event_queue, session_id)

        state_a, state_b = await asyncio.gather(
            a.run(pipeline_state),
            b.run(pipeline_state),
        )

        # Merge findings — researcher_b results override on same task id
        merged = state_a
        for task_b in state_b.sub_tasks:
            for i, task_a in enumerate(merged.sub_tasks):
                if task_a.id == task_b.id:
                    merged.sub_tasks[i] = task_b
                    break

        merged.agent_traces.update(state_b.agent_traces)
        return {**state, "pipeline": merged.model_dump(mode="json")}

    def should_refine(state: GraphState) -> Literal["refine", "synthesize"]:
        """Routing function: check if Critic requested refinement."""
        pipeline = PipelineState(**state["pipeline"])
        critique = pipeline.metadata.get("critique", {})
        needs_refine = critique.get("refinement_needed", False)
        under_budget = pipeline.refinement_count < 2  # hard cap

        if needs_refine and under_budget:
            log.info("graph.routing", decision="refine",
                     iteration=pipeline.refinement_count)
            return "refine"
        return "synthesize"

    # Build the graph
    graph = StateGraph(GraphState)

    graph.add_node("orchestrator", make_node(OrchestratorAgent))
    graph.add_node("planner", make_node(PlannerAgent))
    graph.add_node("research", parallel_research)
    graph.add_node("critic", make_node(CriticAgent))
    graph.add_node("synthesizer", make_node(SynthesizerAgent))

    # Define edges
    graph.set_entry_point("orchestrator")
    graph.add_edge("orchestrator", "planner")
    graph.add_edge("planner", "research")
    graph.add_edge("research", "critic")
    graph.add_conditional_edges(
        "critic",
        should_refine,
        {"refine": "research", "synthesize": "synthesizer"},
    )
    graph.add_edge("synthesizer", END)

    # In production: swap MemorySaver → PostgresSaver with connection pool
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)


class ResearchPipeline:
    """
    High-level interface to the compiled graph.
    Manages lifecycle, metrics, and SSE event emission.
    """

    def __init__(self, session_id: str, event_queue: asyncio.Queue[SSEEvent]):
        self.session_id = session_id
        self.queue = event_queue
        self.budget = TokenBudget(max_tokens=800_000)

    async def run(self, state: PipelineState) -> PipelineState:
        import time
        pipeline_active_gauge.inc()
        start = time.monotonic()

        await self.queue.put(SSEEvent(
            event=SSEEventType.PIPELINE_START,
            session_id=self.session_id,
            data={
                "query": state.query,
                "session_id": self.session_id,
                "message": "Research pipeline initializing",
            }
        ))

        graph = build_pipeline(self.queue, self.session_id, self.budget)
        config = {"configurable": {"thread_id": self.session_id}}

        try:
            initial_graph_state: GraphState = {
                "pipeline": state.model_dump(mode="json"),
                "session_id": self.session_id,
                "iteration": 0,
            }
            final_graph_state = await graph.ainvoke(initial_graph_state, config)
            final_state = PipelineState(**final_graph_state["pipeline"])

            duration = time.monotonic() - start
            pipeline_runs_total.labels(status="success").inc()
            pipeline_duration_seconds.observe(duration)

            await self.queue.put(SSEEvent(
                event=SSEEventType.PIPELINE_COMPLETE,
                session_id=self.session_id,
                data={
                    "session_id": self.session_id,
                    "duration_seconds": round(duration, 2),
                    "tokens_used": self.budget.used,
                    "report_sections": len(final_state.report.sections) if final_state.report else 0,
                    "total_sources": final_state.report.total_sources if final_state.report else 0,
                    "overall_confidence": final_state.report.overall_confidence if final_state.report else 0,
                }
            ))
            return final_state

        except Exception as exc:
            duration = time.monotonic() - start
            pipeline_runs_total.labels(status="error").inc()
            log.error("pipeline.failed", error=str(exc), session_id=self.session_id)
            state.status = ResearchStatus.FAILED
            state.error_message = str(exc)

            await self.queue.put(SSEEvent(
                event=SSEEventType.PIPELINE_ERROR,
                session_id=self.session_id,
                data={"error": str(exc), "duration_seconds": round(duration, 2)}
            ))
            return state

        finally:
            pipeline_active_gauge.dec()
