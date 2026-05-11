"""
Research API endpoints:
  POST /research           — start a new research pipeline
  GET  /research/{id}/stream — SSE stream of live agent events
  GET  /research/{id}      — get completed report (polling fallback)
  GET  /research/history   — user session history
"""
from __future__ import annotations
import asyncio
from uuid import uuid4

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.core.graph.pipeline import ResearchPipeline
from app.core.memory.session import (
    cleanup_session,
    create_session_queue,
    get_session_queue,
    load_pipeline_state,
    save_pipeline_state,
    append_session_to_history,
)
from app.schemas.research import (
    PipelineState,
    ResearchRequest,
    ResearchResponse,
    ResearchStatus,
    SSEEvent,
    SSEEventType,
)

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/research", tags=["research"])

# Active pipeline tasks (session_id → asyncio.Task)
_active_pipelines: dict[str, asyncio.Task] = {}  # type: ignore[type-arg]


@router.post("", response_model=ResearchResponse, status_code=202)
async def start_research(
    request: ResearchRequest,
    background_tasks: BackgroundTasks,
) -> ResearchResponse:
    """
    Start a new research pipeline. Returns a session_id immediately.
    The client should connect to /research/{session_id}/stream for live updates.
    """
    session_id = str(uuid4())

    initial_state = PipelineState(
        session_id=session_id,  # type: ignore[arg-type]
        query=request.query,
        status=ResearchStatus.QUEUED,
        metadata={
            "depth": request.depth,
            "max_sources": request.max_sources,
            "include_arxiv": request.include_arxiv,
            "language": request.language,
        },
    )

    # Persist initial state
    await save_pipeline_state(session_id, initial_state)

    # Create SSE queue before launching the task
    queue = await create_session_queue(session_id)

    # Launch pipeline as background task
    task = asyncio.create_task(
        _run_pipeline(session_id, initial_state, queue),
        name=f"pipeline-{session_id}",
    )
    _active_pipelines[session_id] = task

    log.info("research.started",
             session_id=session_id,
             query=request.query[:80],
             depth=request.depth)

    return ResearchResponse(
        session_id=session_id,
        status=ResearchStatus.QUEUED,
        message="Research pipeline queued. Connect to /research/{session_id}/stream for live updates.",
    )


@router.get("/{session_id}/stream")
async def stream_research(
    session_id: str,
    request: Request,
) -> StreamingResponse:
    """
    SSE endpoint — streams live agent events as they occur.
    Client receives typed events: agent_start, agent_progress, sources_found,
    refinement_loop, report_section, pipeline_complete, etc.
    """
    queue = get_session_queue(session_id)
    if queue is None:
        # Session may be complete — check Redis
        state = await load_pipeline_state(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Session not found")

        # Session exists but queue is gone — stream the completed report
        async def completed_stream():
            if state.report:
                event = SSEEvent(
                    event=SSEEventType.PIPELINE_COMPLETE,
                    session_id=session_id,
                    data={
                        "report": state.report.model_dump(mode="json"),
                        "status": "complete",
                    }
                )
                yield event.to_sse_string()
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            completed_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    async def event_generator():
        heartbeat_interval = 25  # seconds
        last_heartbeat = asyncio.get_event_loop().time()

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    log.info("sse.client_disconnected", session_id=session_id)
                    break

                now = asyncio.get_event_loop().time()

                try:
                    # Wait for next event with timeout for heartbeats
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    yield event.to_sse_string()

                    # Check for terminal events
                    if event.event in (
                        SSEEventType.PIPELINE_COMPLETE,
                        SSEEventType.PIPELINE_ERROR,
                    ):
                        # Attach full report to complete event if available
                        if event.event == SSEEventType.PIPELINE_COMPLETE:
                            state = await load_pipeline_state(session_id)
                            if state and state.report:
                                report_event = SSEEvent(
                                    event=SSEEventType.PIPELINE_COMPLETE,
                                    session_id=session_id,
                                    data={
                                        **event.data,
                                        "report": state.report.model_dump(mode="json"),
                                    }
                                )
                                yield report_event.to_sse_string()
                        yield "event: done\ndata: {}\n\n"
                        break

                except asyncio.TimeoutError:
                    pass

                # Send heartbeat
                if now - last_heartbeat >= heartbeat_interval:
                    hb = SSEEvent(
                        event=SSEEventType.HEARTBEAT,
                        session_id=session_id,
                        data={"alive": True},
                    )
                    yield hb.to_sse_string()
                    last_heartbeat = now

        except asyncio.CancelledError:
            log.info("sse.cancelled", session_id=session_id)
        finally:
            # Only clean up if pipeline is done
            state = await load_pipeline_state(session_id)
            if state and state.status in (ResearchStatus.COMPLETE, ResearchStatus.FAILED):
                await cleanup_session(session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


@router.get("/{session_id}")
async def get_research(session_id: str):
    """Polling fallback — get current pipeline state and report."""
    state = await load_pipeline_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


async def _run_pipeline(
    session_id: str,
    state: PipelineState,
    queue: asyncio.Queue[SSEEvent],
) -> None:
    """Background task that runs the full pipeline and persists results."""
    import structlog.contextvars
    structlog.contextvars.bind_contextvars(session_id=session_id)

    try:
        pipeline = ResearchPipeline(session_id=session_id, event_queue=queue)
        final_state = await pipeline.run(state)
        await save_pipeline_state(session_id, final_state)
        log.info("pipeline.complete",
                 session_id=session_id,
                 sections=len(final_state.report.sections) if final_state.report else 0)
    except Exception as exc:
        log.error("pipeline.background_error", error=str(exc), session_id=session_id)
        state.status = ResearchStatus.FAILED
        state.error_message = str(exc)
        await save_pipeline_state(session_id, state)
    finally:
        _active_pipelines.pop(session_id, None)
        structlog.contextvars.unbind_contextvars("session_id")
