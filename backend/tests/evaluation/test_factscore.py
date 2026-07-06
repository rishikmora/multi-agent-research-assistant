"""
Tests for FActScoreBenchmark (Week 4, System 1).
Run: pytest tests/evaluation/test_factscore.py -v
"""
import json
from unittest.mock import AsyncMock, patch

import pytest

from app.evaluation.factscore import (
    AtomicFactDecomposer,
    FactVerifier,
    FActScoreBenchmark,
    FActScoreError,
    FactSupportVerdict,
    MIN_TEXT_LENGTH_FOR_DECOMPOSITION,
    MAX_ATOMIC_FACTS_PER_BATCH,
)
from app.core.llm_client import TokenBudget


@pytest.fixture
def budget():
    return TokenBudget(max_tokens=200_000)


def decomposition_response(facts: list[str]) -> str:
    return json.dumps([{"fact": f, "source_sentence": f} for f in facts])


def verification_response(verdicts: list[str]) -> str:
    return json.dumps([
        {"index": i, "verdict": v, "supporting_source_ids": ["s1"] if v == "supported" else [], "reasoning": "test"}
        for i, v in enumerate(verdicts)
    ])


class TestAtomicFactDecomposer:
    @pytest.mark.asyncio
    async def test_trivial_text_skipped(self, budget):
        decomposer = AtomicFactDecomposer(budget, "session-1")
        with patch("app.evaluation.factscore.LLMClient.complete", new_callable=AsyncMock) as mock:
            result = await decomposer.decompose("hi")
        mock.assert_not_called()
        assert result == []

    @pytest.mark.asyncio
    async def test_decomposes_compound_sentence_into_multiple_facts(self, budget):
        decomposer = AtomicFactDecomposer(budget, "session-1")
        response = decomposition_response([
            "IBM's Condor was released in 2023",
            "IBM's Condor has 1121 qubits",
        ])
        with patch(
            "app.evaluation.factscore.LLMClient.complete",
            new_callable=AsyncMock, return_value=response,
        ):
            facts = await decomposer.decompose(
                "IBM's Condor, released in 2023, has 1121 qubits and improved error correction."
            )

        assert len(facts) == 2
        assert facts[0].text == "IBM's Condor was released in 2023"

    @pytest.mark.asyncio
    async def test_respects_max_facts_cap(self, budget):
        decomposer = AtomicFactDecomposer(budget, "session-1")
        response = decomposition_response([f"fact {i}" for i in range(50)])
        with patch(
            "app.evaluation.factscore.LLMClient.complete",
            new_callable=AsyncMock, return_value=response,
        ):
            facts = await decomposer.decompose("A" * 100)
        assert len(facts) <= MAX_ATOMIC_FACTS_PER_BATCH

    @pytest.mark.asyncio
    async def test_parse_failure_returns_empty_not_crash(self, budget):
        decomposer = AtomicFactDecomposer(budget, "session-1")
        with patch(
            "app.evaluation.factscore.LLMClient.complete",
            new_callable=AsyncMock, return_value="not json {{{",
        ):
            facts = await decomposer.decompose("A" * 100)
        assert facts == []

    @pytest.mark.asyncio
    async def test_llm_hard_failure_raises(self, budget):
        decomposer = AtomicFactDecomposer(budget, "session-1")
        with patch(
            "app.evaluation.factscore.LLMClient.complete",
            new_callable=AsyncMock, side_effect=RuntimeError("timeout"),
        ):
            with pytest.raises(FActScoreError):
                await decomposer.decompose("A" * 100)


class TestFactVerifier:
    @pytest.mark.asyncio
    async def test_no_sources_marks_all_unsupported(self, budget):
        from app.evaluation.factscore import AtomicFact
        verifier = FactVerifier(budget, "session-1")
        facts = [AtomicFact(text="some claim")]

        result = await verifier.verify_batch(facts, sources=[])

        assert result[0].verdict == FactSupportVerdict.NOT_SUPPORTED
        assert "No sources" in result[0].verification_reasoning

    @pytest.mark.asyncio
    async def test_empty_facts_returns_empty(self, budget):
        verifier = FactVerifier(budget, "session-1")
        result = await verifier.verify_batch([], sources=[{"id": "s1", "title": "t", "snippet": "s"}])
        assert result == []

    @pytest.mark.asyncio
    async def test_verifies_facts_against_sources(self, budget):
        from app.evaluation.factscore import AtomicFact
        verifier = FactVerifier(budget, "session-1")
        facts = [AtomicFact(text="IBM Condor has 1121 qubits"), AtomicFact(text="Fabricated claim")]
        response = verification_response(["supported", "not_supported"])

        with patch(
            "app.evaluation.factscore.LLMClient.complete",
            new_callable=AsyncMock, return_value=response,
        ):
            result = await verifier.verify_batch(
                facts, sources=[{"id": "s1", "title": "IBM press release", "snippet": "1121 qubits"}]
            )

        assert result[0].verdict == FactSupportVerdict.SUPPORTED
        assert result[1].verdict == FactSupportVerdict.NOT_SUPPORTED

    @pytest.mark.asyncio
    async def test_missing_verdict_defaults_to_not_supported(self, budget):
        from app.evaluation.factscore import AtomicFact
        verifier = FactVerifier(budget, "session-1")
        facts = [AtomicFact(text="claim A"), AtomicFact(text="claim B")]
        # Response only covers index 0, not 1
        response = json.dumps([{"index": 0, "verdict": "supported", "reasoning": "ok"}])

        with patch(
            "app.evaluation.factscore.LLMClient.complete",
            new_callable=AsyncMock, return_value=response,
        ):
            result = await verifier.verify_batch(facts, sources=[{"id": "s1", "title": "t", "snippet": "s"}])

        assert result[1].verdict == FactSupportVerdict.NOT_SUPPORTED

    @pytest.mark.asyncio
    async def test_invalid_verdict_string_defaults_to_not_supported(self, budget):
        from app.evaluation.factscore import AtomicFact
        verifier = FactVerifier(budget, "session-1")
        facts = [AtomicFact(text="claim")]
        response = json.dumps([{"index": 0, "verdict": "maybe_kinda", "reasoning": "unclear"}])

        with patch(
            "app.evaluation.factscore.LLMClient.complete",
            new_callable=AsyncMock, return_value=response,
        ):
            result = await verifier.verify_batch(facts, sources=[{"id": "s1", "title": "t", "snippet": "s"}])

        assert result[0].verdict == FactSupportVerdict.NOT_SUPPORTED


class TestFActScoreBenchmark:
    @pytest.mark.asyncio
    async def test_full_evaluation_computes_correct_score(self, budget):
        benchmark = FActScoreBenchmark(budget, "session-1")

        decomp_response = decomposition_response(["fact A", "fact B", "fact C", "fact D"])
        verify_response = verification_response(["supported", "supported", "not_supported", "supported"])

        call_sequence = iter([decomp_response, verify_response])
        async def sequenced(**kwargs):
            return next(call_sequence)

        with patch("app.evaluation.factscore.LLMClient.complete", side_effect=sequenced):
            result = await benchmark.evaluate(
                "Some report text with multiple claims in it that is long enough",
                sources=[{"id": "s1", "title": "t", "snippet": "s"}],
            )

        assert result.total_atomic_facts == 4
        assert result.supported_facts == 3
        assert result.not_supported_facts == 1
        assert result.score == pytest.approx(0.75, abs=0.01)

    @pytest.mark.asyncio
    async def test_irrelevant_facts_excluded_from_score_denominator(self, budget):
        benchmark = FActScoreBenchmark(budget, "session-1")

        decomp_response = decomposition_response(["fact A", "fact B", "fact C"])
        verify_response = verification_response(["supported", "not_supported", "irrelevant"])

        call_sequence = iter([decomp_response, verify_response])
        async def sequenced(**kwargs):
            return next(call_sequence)

        with patch("app.evaluation.factscore.LLMClient.complete", side_effect=sequenced):
            result = await benchmark.evaluate(
                "Report text long enough to trigger decomposition properly here",
                sources=[{"id": "s1", "title": "t", "snippet": "s"}],
            )

        # Score = supported / (supported + not_supported), irrelevant excluded
        assert result.score == pytest.approx(0.5, abs=0.01)
        assert result.irrelevant_facts == 1

    @pytest.mark.asyncio
    async def test_trivial_text_returns_zero_result(self, budget):
        benchmark = FActScoreBenchmark(budget, "session-1")
        result = await benchmark.evaluate("hi", sources=[])
        assert result.total_atomic_facts == 0

    @pytest.mark.asyncio
    async def test_flagged_facts_property_returns_only_unsupported(self, budget):
        benchmark = FActScoreBenchmark(budget, "session-1")

        decomp_response = decomposition_response(["fact A", "fact B"])
        verify_response = verification_response(["supported", "not_supported"])

        call_sequence = iter([decomp_response, verify_response])
        async def sequenced(**kwargs):
            return next(call_sequence)

        with patch("app.evaluation.factscore.LLMClient.complete", side_effect=sequenced):
            result = await benchmark.evaluate(
                "Report text long enough to trigger decomposition properly here",
                sources=[{"id": "s1", "title": "t", "snippet": "s"}],
            )

        assert len(result.flagged_facts) == 1
        assert result.flagged_facts[0].text == "fact B"

    @pytest.mark.asyncio
    async def test_to_dict_json_serializable(self, budget):
        import json as _json
        benchmark = FActScoreBenchmark(budget, "session-1")

        decomp_response = decomposition_response(["fact A"])
        verify_response = verification_response(["supported"])
        call_sequence = iter([decomp_response, verify_response])
        async def sequenced(**kwargs):
            return next(call_sequence)

        with patch("app.evaluation.factscore.LLMClient.complete", side_effect=sequenced):
            result = await benchmark.evaluate(
                "Report text long enough to trigger decomposition properly here",
                sources=[{"id": "s1", "title": "t", "snippet": "s"}],
            )

        _json.dumps(result.to_dict())
