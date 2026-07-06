"""
Tests for MultiHopQueryDecomposer (Week 3, System 3).
Run: pytest tests/retrieval/test_decomposer.py -v
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.retrieval.decomposer import (
    MultiHopQueryDecomposer,
    MIN_SUB_QUERIES,
    MAX_SUB_QUERIES,
    MIN_QUERY_WORDS_FOR_DECOMPOSITION,
)
from app.core.llm_client import TokenBudget


@pytest.fixture
def budget():
    return TokenBudget(max_tokens=100_000)


def decomposition_json(n_subs: int = 4) -> str:
    return json.dumps({
        "needs_decomposition": True,
        "reasoning": "Query bundles multiple distinct facets",
        "sub_queries": [
            {"text": f"sub-query facet {i}", "facet": f"facet_{i}", "priority": i + 1}
            for i in range(n_subs)
        ],
    })


class TestSimpleQueryBypass:
    @pytest.mark.asyncio
    async def test_short_query_skips_decomposition_without_llm_call(self, budget):
        decomposer = MultiHopQueryDecomposer(budget, "session-1")
        with patch("app.retrieval.decomposer.LLMClient.complete", new_callable=AsyncMock) as mock:
            result = await decomposer.decompose("quantum computing")

        mock.assert_not_called()
        assert result.was_decomposed is False
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0].text == "quantum computing"

    @pytest.mark.asyncio
    async def test_force_flag_bypasses_word_count_check(self, budget):
        decomposer = MultiHopQueryDecomposer(budget, "session-1")
        with patch(
            "app.retrieval.decomposer.LLMClient.complete",
            new_callable=AsyncMock,
            return_value=decomposition_json(3),
        ) as mock:
            await decomposer.decompose("short query", force=True)
        mock.assert_called_once()


class TestDecomposition:
    @pytest.mark.asyncio
    async def test_complex_query_decomposed_into_multiple_subqueries(self, budget):
        decomposer = MultiHopQueryDecomposer(budget, "session-1")
        with patch(
            "app.retrieval.decomposer.LLMClient.complete",
            new_callable=AsyncMock,
            return_value=decomposition_json(4),
        ):
            result = await decomposer.decompose(
                "Impact of AI on semiconductor manufacturing supply chains globally"
            )

        assert result.was_decomposed is True
        assert len(result.sub_queries) == 4
        assert all(sq.facet.startswith("facet_") for sq in result.sub_queries)

    @pytest.mark.asyncio
    async def test_respects_max_sub_queries_cap(self, budget):
        decomposer = MultiHopQueryDecomposer(budget, "session-1")
        with patch(
            "app.retrieval.decomposer.LLMClient.complete",
            new_callable=AsyncMock,
            return_value=decomposition_json(10),   # LLM returns way more than allowed
        ):
            result = await decomposer.decompose(
                "A very long and complex multi-part research query about many things"
            )

        assert len(result.sub_queries) <= MAX_SUB_QUERIES

    @pytest.mark.asyncio
    async def test_llm_says_no_decomposition_needed_respected(self, budget):
        decomposer = MultiHopQueryDecomposer(budget, "session-1")
        response = json.dumps({
            "needs_decomposition": False,
            "reasoning": "Already narrow and singular",
            "sub_queries": [],
        })
        with patch(
            "app.retrieval.decomposer.LLMClient.complete",
            new_callable=AsyncMock,
            return_value=response,
        ):
            result = await decomposer.decompose(
                "What year was the transformer architecture paper published"
            )

        assert result.was_decomposed is False
        assert len(result.sub_queries) == 1


class TestFailureHandling:
    @pytest.mark.asyncio
    async def test_parse_failure_falls_back_to_direct_query(self, budget):
        decomposer = MultiHopQueryDecomposer(budget, "session-1")
        with patch(
            "app.retrieval.decomposer.LLMClient.complete",
            new_callable=AsyncMock,
            return_value="not valid json {{{",
        ):
            result = await decomposer.decompose(
                "A sufficiently long query to trigger decomposition attempt here"
            )

        assert result.was_decomposed is False
        assert len(result.sub_queries) == 1
        assert result.sub_queries[0].text == (
            "A sufficiently long query to trigger decomposition attempt here"
        )

    @pytest.mark.asyncio
    async def test_llm_failure_raises_decomposer_error(self, budget):
        from app.retrieval.decomposer import QueryDecomposerError
        decomposer = MultiHopQueryDecomposer(budget, "session-1")
        with patch(
            "app.retrieval.decomposer.LLMClient.complete",
            new_callable=AsyncMock,
            side_effect=RuntimeError("API timeout"),
        ):
            with pytest.raises(QueryDecomposerError):
                await decomposer.decompose(
                    "A sufficiently long query to trigger a real decomposition attempt"
                )

    @pytest.mark.asyncio
    async def test_below_min_subqueries_padded_with_original(self, budget):
        decomposer = MultiHopQueryDecomposer(budget, "session-1")
        response = json.dumps({
            "needs_decomposition": True,
            "reasoning": "test",
            "sub_queries": [{"text": "only one facet", "facet": "solo", "priority": 1}],
        })
        with patch(
            "app.retrieval.decomposer.LLMClient.complete",
            new_callable=AsyncMock,
            return_value=response,
        ):
            result = await decomposer.decompose(
                "A sufficiently long query needing at least minimum sub queries"
            )

        assert len(result.sub_queries) >= MIN_SUB_QUERIES


class TestSerialization:
    def test_to_dict_json_serializable(self):
        import json as _json
        from app.retrieval.decomposer import DecompositionResult, SubQuery
        result = DecompositionResult(
            original_query="test",
            sub_queries=[SubQuery(text="a", facet="b", priority=1)],
            was_decomposed=True,
        )
        _json.dumps(result.to_dict())
