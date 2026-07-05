"""
Memory system API endpoints.

GET  /memory/health   — dashboard stats across tiers 2 and 3
GET  /memory/preview   — dry-run retrieval, shows what context a query
                          would get without running a full pipeline
GET  /memory/conflicts  — list pending semantic conflicts
POST /memory/conflicts/{id}/resolve — Critic/human resolves a conflict
"""
from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session   # existing dependency in the codebase
from app.memory.retriever import MemoryRetriever

router = APIRouter(prefix="/memory", tags=["memory"])


class ConflictResolutionRequest(BaseModel):
    resolution: str
    winning_claim: str | None = None


@router.get("/health")
async def memory_health(db: AsyncSession = Depends(get_db_session)):
    """
    Combined health snapshot across episodic (tier 2) and semantic
    (tier 3) memory. Polled by the frontend MemoryPanel every 15s.
    """
    retriever = MemoryRetriever(db)
    return await retriever.get_memory_health()


@router.get("/preview")
async def preview_retrieval(
    query: str = Query(..., min_length=3, max_length=500),
    domain: str | None = None,
    db: AsyncSession = Depends(get_db_session),
):
    """
    Dry-run memory retrieval for a query, WITHOUT starting a research
    pipeline. Used by the frontend's "test memory retrieval" widget so
    users (and you, demoing this) can see exactly what context memory
    would inject before committing to a full run.
    """
    retriever = MemoryRetriever(db)
    context = await retriever.retrieve_context(query=query, domain=domain)
    return {
        "formatted_context": context.formatted_context,
        "episodic_hits": context.episodic_hits,
        "semantic_hits": context.semantic_hits,
        "has_content": context.has_content,
        "retrieval_latency_ms": context.retrieval_latency_ms,
    }


@router.get("/conflicts")
async def list_conflicts(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db_session),
):
    retriever = MemoryRetriever(db)
    return await retriever.semantic.get_pending_conflicts(limit=limit)


@router.post("/conflicts/{conflict_id}/resolve")
async def resolve_conflict(
    conflict_id: UUID,
    request: ConflictResolutionRequest,
    db: AsyncSession = Depends(get_db_session),
):
    retriever = MemoryRetriever(db)
    try:
        await retriever.semantic.resolve_conflict(
            conflict_id=conflict_id,
            resolution=request.resolution,
            winning_claim=request.winning_claim,
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"status": "resolved", "conflict_id": str(conflict_id)}


@router.get("/user/{user_id}/history")
async def user_research_history(
    user_id: UUID,
    limit: int = Query(20, ge=1, le=50),
    db: AsyncSession = Depends(get_db_session),
):
    """A user's past research sessions — the raw episodic history."""
    retriever = MemoryRetriever(db)
    return await retriever.episodic.get_user_history(user_id, limit=limit)
