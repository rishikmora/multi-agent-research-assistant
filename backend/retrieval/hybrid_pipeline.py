"""
Cross-Encoder Reranking + Hybrid Pipeline Orchestrator — Week 3, Systems 4-5

RERANKING: THE HIGHEST-ROI STEP IN THE ENTIRE PIPELINE
BM25 and dense retrieval are both bi-encoders — they encode queries and
documents SEPARATELY, then compare the resulting vectors/scores. This is
fast (required for scanning thousands of candidates) but throws away
cross-attention between query and document tokens. A cross-encoder scores
the (query, document) PAIR jointly, capturing interactions bi-encoders
structurally cannot — at the cost of being too slow to run over a full
corpus. The standard production pattern: bi-encoders (BM25 + dense) do
first-stage retrieval over the full pool; a cross-encoder reranks only
the small fused candidate set (typically top 20-30) that survives fusion.
One additional model call on a small candidate set delivers the largest
single accuracy jump available in the retrieval stack.

THIS IMPLEMENTATION uses an LLM-as-cross-encoder rather than a dedicated
reranking model (e.g. Cohere Rerank, a fine-tuned MiniLM cross-encoder).
Tradeoff, stated plainly: an LLM call is slower and more expensive per
candidate batch than a purpose-built reranker model, but requires no
additional model hosting/serving infrastructure and integrates directly
with the LLMClient abstraction already used throughout MARS. The
interface (`rerank(query, candidates) -> ranked candidates`) is designed
so a dedicated reranker model can be substituted later without touching
any caller.

THE FULL PIPELINE (this is what Researcher agents actually call):
  1. MultiHopQueryDecomposer  → 1 query becomes 3-5 targeted sub-queries
  2. Per sub-query, in parallel: BM25Index.search() + dense_search_fn()
  3. ReciprocalRankFusion.fuse() → merge all ranked lists into one
  4. CrossEncoderReranker.rerank() → refine the top candidates
  5. Return final ranked Source list to the calling agent
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import structlog

from app.core.llm_client import LLMClient, TokenBudget
from app.retrieval.bm25 import BM25Index
from app.retrieval.rrf import ReciprocalRankFusion, RankedList, FusedResult
from app.retrieval.decomposer import MultiHopQueryDecomposer, DecompositionResult

log = structlog.get_logger(__name__)

RERANK_BATCH_SIZE = 25       # Cap on candidates sent to the reranker per call
DEFAULT_TOP_K_FINAL = 10


@dataclass
class RetrievalCandidate:
    """A document from either BM25 or dense retrieval, pre-fusion."""
    doc_id: str
    title: str
    snippet: str
    url: str
    source_type: str = "web"

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id, "title": self.title,
            "snippet": self.snippet, "url": self.url, "source_type": self.source_type,
        }


@dataclass
class RerankedResult:
    doc_id: str
    rerank_score: float
    original_rrf_rank: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "rerank_score": round(self.rerank_score, 4),
            "original_rrf_rank": self.original_rrf_rank,
        }


@dataclass
class HybridRetrievalResult:
    query: str
    sub_queries_used: list[str]
    candidates: list[RetrievalCandidate]     # Final, reranked, in order
    fusion_stats: dict[str, Any] = field(default_factory=dict)
    reranked: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "sub_queries_used": self.sub_queries_used,
            "candidates": [c.to_dict() for c in self.candidates],
            "fusion_stats": self.fusion_stats,
            "reranked": self.reranked,
        }


class RerankerError(Exception):
    pass


class CrossEncoderReranker:
    """
    LLM-as-cross-encoder. See module docstring for the tradeoff vs a
    dedicated reranker model.
    """

    def __init__(self, budget: TokenBudget, session_id: str):
        self.llm = LLMClient("critic", budget)
        self.session_id = session_id

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int = DEFAULT_TOP_K_FINAL,
    ) -> list[RerankedResult]:
        if not candidates:
            return []

        batch = candidates[:RERANK_BATCH_SIZE]
        candidate_block = "\n".join(
            f"[{i}] {c.title}: {c.snippet[:200]}"
            for i, c in enumerate(batch)
        )

        system = (
            "You are a relevance reranker. Score each candidate's relevance "
            "to the query on a 0-100 scale, judging true topical relevance "
            "and specificity, not just keyword overlap. Return ONLY valid JSON."
        )
        prompt = f"""Query: {query}

Candidates:
{candidate_block}

Return JSON array of {{"index": i, "relevance": 0-100}} for ALL {len(batch)} candidates,
ordered by index. Judge genuine relevance to answering the query, not
superficial keyword matches."""

        try:
            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=1024,
                temperature=0.0,
                session_id=self.session_id,
            )
            scored_raw = json.loads(response.strip())
        except json.JSONDecodeError as exc:
            log.warning("reranker.parse_failed", error=str(exc))
            # Fail safe: preserve original fused order rather than crash
            return [
                RerankedResult(doc_id=c.doc_id, rerank_score=1.0 - (i * 0.01), original_rrf_rank=i + 1)
                for i, c in enumerate(batch[:top_k])
            ]
        except Exception as exc:
            log.error("reranker.llm_failed", error=str(exc))
            raise RerankerError(f"Reranking failed: {exc}") from exc

        scores: dict[int, float] = {}
        for item in scored_raw:
            if isinstance(item, dict) and "index" in item:
                idx = item["index"]
                if 0 <= idx < len(batch):
                    scores[idx] = float(item.get("relevance", 0))

        results = [
            RerankedResult(
                doc_id=batch[i].doc_id,
                rerank_score=scores.get(i, 0.0),
                original_rrf_rank=i + 1,
            )
            for i in range(len(batch))
        ]
        results.sort(key=lambda r: r.rerank_score, reverse=True)

        log.info("reranker.complete", n_candidates=len(batch), top_score=results[0].rerank_score if results else None)
        return results[:top_k]


class HybridRetrievalPipeline:
    """
    The full pipeline orchestrator — this is what a Researcher agent calls
    instead of a raw web_search().

    Usage:
        pipeline = HybridRetrievalPipeline(budget, session_id, web_search_fn, dense_search_fn)
        result = await pipeline.retrieve("Impact of AI on semiconductor manufacturing")
    """

    def __init__(
        self,
        budget: TokenBudget,
        session_id: str,
        web_search_fn: Callable[[str], Awaitable[list[RetrievalCandidate]]],
        dense_search_fn: Callable[[str], Awaitable[list[RetrievalCandidate]]] | None = None,
        rrf_k: int | None = None,
    ):
        self.budget = budget
        self.session_id = session_id
        self.web_search_fn = web_search_fn
        self.dense_search_fn = dense_search_fn
        self.decomposer = MultiHopQueryDecomposer(budget, session_id)
        self.rrf = ReciprocalRankFusion(k=rrf_k)
        self.reranker = CrossEncoderReranker(budget, session_id)

    async def retrieve(
        self,
        query: str,
        domain: str = "general",
        top_k_final: int = DEFAULT_TOP_K_FINAL,
        skip_rerank: bool = False,
    ) -> HybridRetrievalResult:
        decomposition = await self.decomposer.decompose(query, domain=domain)
        sub_query_texts = [sq.text for sq in decomposition.sub_queries]

        # Parallel retrieval across all sub-queries — never sequential,
        # see module docstring in decomposer.py for why order-independence matters
        async def retrieve_for_subquery(sub_query: str) -> tuple[list[RetrievalCandidate], list[RetrievalCandidate]]:
            web_task = self.web_search_fn(sub_query)
            dense_task = self.dense_search_fn(sub_query) if self.dense_search_fn else _empty_list()
            web_results, dense_results = await asyncio.gather(web_task, dense_task, return_exceptions=True)
            web_results = web_results if isinstance(web_results, list) else []
            dense_results = dense_results if isinstance(dense_results, list) else []
            return web_results, dense_results

        all_sub_results = await asyncio.gather(
            *[retrieve_for_subquery(sq) for sq in sub_query_texts],
            return_exceptions=True,
        )

        # Aggregate all candidates across all sub-queries, deduplicating by doc_id
        candidate_pool: dict[str, RetrievalCandidate] = {}
        web_ranked_ids: list[str] = []
        dense_ranked_ids: list[str] = []

        for result in all_sub_results:
            if isinstance(result, Exception):
                log.warning("hybrid_retrieval.subquery_failed", error=str(result))
                continue
            web_results, dense_results = result

            for c in web_results:
                candidate_pool.setdefault(c.doc_id, c)
                if c.doc_id not in web_ranked_ids:
                    web_ranked_ids.append(c.doc_id)
            for c in dense_results:
                candidate_pool.setdefault(c.doc_id, c)
                if c.doc_id not in dense_ranked_ids:
                    dense_ranked_ids.append(c.doc_id)

        if not candidate_pool:
            log.warning("hybrid_retrieval.no_candidates", query=query[:60])
            return HybridRetrievalResult(
                query=query, sub_queries_used=sub_query_texts,
                candidates=[], fusion_stats={"total_candidates": 0},
            )

        # BM25 pass over the full aggregated candidate pool for THIS query
        bm25_index = BM25Index()
        bm25_index.add_documents([
            (c.doc_id, f"{c.title} {c.snippet}") for c in candidate_pool.values()
        ])
        bm25_results = bm25_index.search(query, top_k=len(candidate_pool))
        bm25_ranked_ids = [r.doc_id for r in bm25_results]

        # RRF fusion across BM25 (this query) + web ranking + dense ranking
        ranked_lists = [RankedList("bm25", bm25_ranked_ids)]
        if web_ranked_ids:
            ranked_lists.append(RankedList("web_search", web_ranked_ids))
        if dense_ranked_ids:
            ranked_lists.append(RankedList("dense", dense_ranked_ids))

        fused = self.rrf.fuse(ranked_lists)
        fused_candidates = [
            candidate_pool[f.doc_id] for f in fused if f.doc_id in candidate_pool
        ]

        fusion_stats = {
            "total_candidates": len(candidate_pool),
            "sub_queries": len(sub_query_texts),
            "rankers_used": [rl.ranker_name for rl in ranked_lists],
            "fused_count": len(fused),
        }

        if skip_rerank:
            final = fused_candidates[:top_k_final]
            return HybridRetrievalResult(
                query=query, sub_queries_used=sub_query_texts,
                candidates=final, fusion_stats=fusion_stats, reranked=False,
            )

        # Cross-encoder rerank on the top of the fused list
        reranked = await self.reranker.rerank(
            query, fused_candidates[:RERANK_BATCH_SIZE], top_k=top_k_final
        )
        reranked_ids = {r.doc_id for r in reranked}
        final_candidates = [
            candidate_pool[r.doc_id] for r in reranked if r.doc_id in candidate_pool
        ]

        log.info("hybrid_retrieval.pipeline_complete",
                query=query[:60], final_count=len(final_candidates),
                **fusion_stats)

        return HybridRetrievalResult(
            query=query,
            sub_queries_used=sub_query_texts,
            candidates=final_candidates,
            fusion_stats=fusion_stats,
            reranked=True,
        )


async def _empty_list() -> list[RetrievalCandidate]:
    return []
