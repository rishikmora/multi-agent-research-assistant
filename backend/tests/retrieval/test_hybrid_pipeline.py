"""
Tests for CrossEncoderReranker and HybridRetrievalPipeline (Week 3, Systems 4-5).
Run: pytest tests/retrieval/test_hybrid_pipeline.py -v
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.retrieval.hybrid_pipeline import (
    CrossEncoderReranker,
    HybridRetrievalPipeline,
    RetrievalCandidate,
    RerankerError,
)
from app.core.llm_client import TokenBudget


@pytest.fixture
def budget():
    return TokenBudget(max_tokens=200_000)


def make_candidates(n: int) -> list[RetrievalCandidate]:
    return [
        RetrievalCandidate(
            doc_id=f"doc{i}", title=f"Title {i}", snippet=f"Snippet content {i}",
            url=f"https://example.com/{i}",
        )
        for i in range(n)
    ]


def rerank_response_json(indices_scores: list[tuple[int, int]]) -> str:
    return json.dumps([{"index": i, "relevance": s} for i, s in indices_scores])


class TestCrossEncoderReranker:
    @pytest.mark.asyncio
    async def test_empty_candidates_returns_empty(self, budget):
        reranker = CrossEncoderReranker(budget, "session-1")
        result = await reranker.rerank("query", [])
        assert result == []

    @pytest.mark.asyncio
    async def test_reranks_by_relevance_score(self, budget):
        candidates = make_candidates(3)
        response = rerank_response_json([(0, 30), (1, 90), (2, 60)])
        with patch(
            "app.retrieval.hybrid_pipeline.LLMClient.complete",
            new_callable=AsyncMock, return_value=response,
        ):
            reranker = CrossEncoderReranker(budget, "session-1")
            result = await reranker.rerank("query", candidates, top_k=3)

        assert result[0].doc_id == "doc1"   # Highest relevance score
        assert result[1].doc_id == "doc2"
        assert result[2].doc_id == "doc0"

    @pytest.mark.asyncio
    async def test_respects_top_k(self, budget):
        candidates = make_candidates(10)
        response = rerank_response_json([(i, 100 - i) for i in range(10)])
        with patch(
            "app.retrieval.hybrid_pipeline.LLMClient.complete",
            new_callable=AsyncMock, return_value=response,
        ):
            reranker = CrossEncoderReranker(budget, "session-1")
            result = await reranker.rerank("query", candidates, top_k=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_parse_failure_falls_back_to_original_order(self, budget):
        candidates = make_candidates(3)
        with patch(
            "app.retrieval.hybrid_pipeline.LLMClient.complete",
            new_callable=AsyncMock, return_value="not json {{{",
        ):
            reranker = CrossEncoderReranker(budget, "session-1")
            result = await reranker.rerank("query", candidates, top_k=3)

        # Falls back gracefully, preserving original order, not crashing
        assert len(result) == 3
        assert result[0].doc_id == "doc0"

    @pytest.mark.asyncio
    async def test_llm_hard_failure_raises_reranker_error(self, budget):
        candidates = make_candidates(2)
        with patch(
            "app.retrieval.hybrid_pipeline.LLMClient.complete",
            new_callable=AsyncMock, side_effect=RuntimeError("connection reset"),
        ):
            reranker = CrossEncoderReranker(budget, "session-1")
            with pytest.raises(RerankerError):
                await reranker.rerank("query", candidates)

    @pytest.mark.asyncio
    async def test_batch_size_cap_enforced(self, budget):
        from app.retrieval.hybrid_pipeline import RERANK_BATCH_SIZE
        candidates = make_candidates(RERANK_BATCH_SIZE + 20)
        response = rerank_response_json([(i, 50) for i in range(RERANK_BATCH_SIZE)])

        captured_prompt = {}

        async def capture(**kwargs):
            captured_prompt["messages"] = kwargs.get("messages")
            return response

        with patch("app.retrieval.hybrid_pipeline.LLMClient.complete", side_effect=capture):
            reranker = CrossEncoderReranker(budget, "session-1")
            await reranker.rerank("query", candidates, top_k=5)

        # Only RERANK_BATCH_SIZE candidates should appear in the prompt sent
        prompt_text = str(captured_prompt["messages"])
        assert f"doc{RERANK_BATCH_SIZE + 5}" not in prompt_text


class TestHybridRetrievalPipeline:
    @pytest.mark.asyncio
    async def test_full_pipeline_end_to_end(self, budget):
        candidates_web = make_candidates(5)

        async def web_search_fn(query: str) -> list[RetrievalCandidate]:
            return candidates_web

        decomposition_response = json.dumps({
            "needs_decomposition": False,
            "reasoning": "simple",
            "sub_queries": [{"text": "test query", "facet": "direct", "priority": 1}],
        })
        rerank_response = rerank_response_json([(i, 100 - i * 10) for i in range(5)])

        call_sequence = iter([decomposition_response, rerank_response])

        async def sequenced(**kwargs):
            return next(call_sequence, rerank_response)

        with patch("app.retrieval.decomposer.LLMClient.complete", side_effect=sequenced), \
             patch("app.retrieval.hybrid_pipeline.LLMClient.complete", side_effect=sequenced):
            pipeline = HybridRetrievalPipeline(
                budget, "session-1", web_search_fn=web_search_fn,
            )
            result = await pipeline.retrieve("test query", top_k_final=3)

        assert len(result.candidates) <= 3
        assert result.reranked is True
        assert result.fusion_stats["total_candidates"] == 5

    @pytest.mark.asyncio
    async def test_no_candidates_returns_empty_result_gracefully(self, budget):
        async def empty_search(query: str) -> list[RetrievalCandidate]:
            return []

        decomposition_response = json.dumps({
            "needs_decomposition": False, "reasoning": "simple",
            "sub_queries": [{"text": "query", "facet": "direct", "priority": 1}],
        })

        with patch(
            "app.retrieval.decomposer.LLMClient.complete",
            new_callable=AsyncMock, return_value=decomposition_response,
        ):
            pipeline = HybridRetrievalPipeline(budget, "session-1", web_search_fn=empty_search)
            result = await pipeline.retrieve("query")

        assert result.candidates == []
        assert result.fusion_stats["total_candidates"] == 0

    @pytest.mark.asyncio
    async def test_skip_rerank_returns_fused_order_directly(self, budget):
        candidates = make_candidates(5)

        async def web_search_fn(query: str) -> list[RetrievalCandidate]:
            return candidates

        decomposition_response = json.dumps({
            "needs_decomposition": False, "reasoning": "simple",
            "sub_queries": [{"text": "query", "facet": "direct", "priority": 1}],
        })

        with patch(
            "app.retrieval.decomposer.LLMClient.complete",
            new_callable=AsyncMock, return_value=decomposition_response,
        ):
            pipeline = HybridRetrievalPipeline(budget, "session-1", web_search_fn=web_search_fn)
            result = await pipeline.retrieve("query", skip_rerank=True, top_k_final=3)

        assert result.reranked is False
        assert len(result.candidates) <= 3

    @pytest.mark.asyncio
    async def test_dense_search_failure_does_not_break_pipeline(self, budget):
        """If the dense search function raises, the pipeline should
        continue with web/BM25 results alone rather than crash."""
        candidates = make_candidates(3)

        async def web_search_fn(query: str) -> list[RetrievalCandidate]:
            return candidates

        async def broken_dense_search(query: str) -> list[RetrievalCandidate]:
            raise RuntimeError("vector db unavailable")

        decomposition_response = json.dumps({
            "needs_decomposition": False, "reasoning": "simple",
            "sub_queries": [{"text": "query", "facet": "direct", "priority": 1}],
        })
        rerank_response = rerank_response_json([(i, 50) for i in range(3)])

        call_sequence = iter([decomposition_response, rerank_response])

        async def sequenced(**kwargs):
            return next(call_sequence, rerank_response)

        with patch("app.retrieval.decomposer.LLMClient.complete", side_effect=sequenced), \
             patch("app.retrieval.hybrid_pipeline.LLMClient.complete", side_effect=sequenced):
            pipeline = HybridRetrievalPipeline(
                budget, "session-1",
                web_search_fn=web_search_fn,
                dense_search_fn=broken_dense_search,
            )
            result = await pipeline.retrieve("query")

        # Pipeline completes despite dense search failure
        assert len(result.candidates) > 0

    @pytest.mark.asyncio
    async def test_deduplicates_candidates_across_subqueries(self, budget):
        shared_candidate = RetrievalCandidate(
            doc_id="shared", title="Shared doc", snippet="appears in both",
            url="https://example.com/shared",
        )

        call_count = {"n": 0}
        async def web_search_fn(query: str) -> list[RetrievalCandidate]:
            call_count["n"] += 1
            return [shared_candidate]

        decomposition_response = json.dumps({
            "needs_decomposition": True, "reasoning": "multi-facet",
            "sub_queries": [
                {"text": "facet A", "facet": "a", "priority": 1},
                {"text": "facet B", "facet": "b", "priority": 2},
            ],
        })
        rerank_response = rerank_response_json([(0, 90)])

        call_sequence = iter([decomposition_response, rerank_response])
        async def sequenced(**kwargs):
            return next(call_sequence, rerank_response)

        with patch("app.retrieval.decomposer.LLMClient.complete", side_effect=sequenced), \
             patch("app.retrieval.hybrid_pipeline.LLMClient.complete", side_effect=sequenced):
            pipeline = HybridRetrievalPipeline(budget, "session-1", web_search_fn=web_search_fn)
            result = await pipeline.retrieve(
                "A sufficiently long query triggering real decomposition here"
            )

        assert result.fusion_stats["total_candidates"] == 1   # Deduplicated
