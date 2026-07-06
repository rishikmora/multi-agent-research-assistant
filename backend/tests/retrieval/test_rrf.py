"""
Tests for ReciprocalRankFusion (Week 3, System 2).
Run: pytest tests/retrieval/test_rrf.py -v
"""
import pytest

from app.retrieval.rrf import (
    ReciprocalRankFusion,
    RankedList,
    DEFAULT_RRF_K_SMALL_CORPUS,
    CORPUS_SIZE_THRESHOLD_FOR_LARGE_K,
)


class TestBasicFusion:
    def test_empty_lists_returns_empty(self):
        rrf = ReciprocalRankFusion(k=20)
        assert rrf.fuse([]) == []

    def test_single_list_preserves_order(self):
        rrf = ReciprocalRankFusion(k=20)
        result = rrf.fuse([RankedList("solo", ["a", "b", "c"])])
        assert [r.doc_id for r in result] == ["a", "b", "c"]

    def test_document_appearing_in_both_lists_ranks_higher(self):
        """The core RRF property: a doc that both rankers agree on
        should outrank a doc only one ranker liked."""
        rrf = ReciprocalRankFusion(k=20)
        result = rrf.fuse([
            RankedList("bm25", ["consensus_doc", "bm25_only"]),
            RankedList("dense", ["consensus_doc", "dense_only"]),
        ])
        assert result[0].doc_id == "consensus_doc"

    def test_strong_single_ranker_vote_does_not_guarantee_top_rank(self):
        """A doc ranked #1 by one ranker but absent from the other should
        not automatically outrank a doc both rankers rank moderately —
        this is RRF's core 'refuses to trust a single strong vote' property."""
        rrf = ReciprocalRankFusion(k=20)
        result = rrf.fuse([
            RankedList("bm25", ["solo_top", "x", "y", "z"]),
            RankedList("dense", ["a", "consensus", "b"]),
            RankedList("bm25_variant", ["c", "consensus", "d"]),
        ])
        # consensus appears in 2 lists, solo_top only in 1 — consensus should win
        consensus_score = next(r.rrf_score for r in result if r.doc_id == "consensus")
        solo_score = next(r.rrf_score for r in result if r.doc_id == "solo_top")
        assert consensus_score > solo_score

    def test_exact_score_formula(self):
        """Verify the actual arithmetic: 1/(k+rank) summed across lists."""
        rrf = ReciprocalRankFusion(k=10)
        result = rrf.fuse([RankedList("only", ["doc_a"])])
        expected = 1 / (10 + 1)
        assert result[0].rrf_score == pytest.approx(expected, abs=1e-9)

    def test_multi_list_score_is_additive(self):
        rrf = ReciprocalRankFusion(k=10)
        result = rrf.fuse([
            RankedList("list1", ["doc_a"]),
            RankedList("list2", ["doc_a"]),
        ])
        expected = 1 / (10 + 1) + 1 / (10 + 1)
        assert result[0].rrf_score == pytest.approx(expected, abs=1e-9)


class TestCorpusSizeAwareK:
    def test_small_corpus_uses_small_k_default(self):
        rrf = ReciprocalRankFusion(k=None)   # auto-select
        small_list = RankedList("bm25", [f"doc{i}" for i in range(20)])
        k = rrf._select_k([small_list])
        assert k == DEFAULT_RRF_K_SMALL_CORPUS

    def test_large_corpus_uses_trec_scale_k(self):
        rrf = ReciprocalRankFusion(k=None)
        large_list = RankedList("bm25", [f"doc{i}" for i in range(CORPUS_SIZE_THRESHOLD_FOR_LARGE_K + 10)])
        k = rrf._select_k([large_list])
        assert k == 60

    def test_explicit_k_overrides_auto_selection(self):
        rrf = ReciprocalRankFusion(k=99)
        large_list = RankedList("bm25", [f"doc{i}" for i in range(1000)])
        k = rrf._select_k([large_list])
        assert k == 99

    def test_k_affects_actual_fusion_scores(self):
        small_k_rrf = ReciprocalRankFusion(k=5)
        large_k_rrf = ReciprocalRankFusion(k=100)
        ranked = [RankedList("r", ["doc_a"])]

        small_k_result = small_k_rrf.fuse(ranked)
        large_k_result = large_k_rrf.fuse(ranked)

        # Smaller k produces a larger score for the same rank position
        assert small_k_result[0].rrf_score > large_k_result[0].rrf_score


class TestWeighting:
    def test_zero_weight_ranker_excluded(self):
        rrf = ReciprocalRankFusion(k=20)
        result = rrf.fuse([
            RankedList("trusted", ["doc_a"], weight=1.0),
            RankedList("untrusted", ["doc_b"], weight=0.0),
        ])
        doc_ids = {r.doc_id for r in result}
        assert "doc_b" not in doc_ids

    def test_higher_weight_increases_contribution(self):
        rrf = ReciprocalRankFusion(k=20)
        result_low = rrf.fuse([RankedList("r", ["doc_a"], weight=0.5)])
        result_high = rrf.fuse([RankedList("r", ["doc_a"], weight=2.0)])
        assert result_high[0].rrf_score > result_low[0].rrf_score
        assert result_high[0].rrf_score == pytest.approx(result_low[0].rrf_score * 4, abs=1e-9)


class TestProvenance:
    def test_contributing_rankers_tracked(self):
        rrf = ReciprocalRankFusion(k=20)
        result = rrf.fuse([
            RankedList("bm25", ["doc_a"]),
            RankedList("dense", ["doc_a"]),
        ])
        assert set(result[0].contributing_rankers) == {"bm25", "dense"}

    def test_per_ranker_rank_positions_tracked(self):
        rrf = ReciprocalRankFusion(k=20)
        result = rrf.fuse([
            RankedList("bm25", ["x", "doc_a"]),
            RankedList("dense", ["doc_a", "y"]),
        ])
        doc_a = next(r for r in result if r.doc_id == "doc_a")
        assert doc_a.per_ranker_ranks["bm25"] == 2
        assert doc_a.per_ranker_ranks["dense"] == 1

    def test_explain_produces_readable_breakdown(self):
        rrf = ReciprocalRankFusion(k=20)
        result = rrf.fuse([RankedList("bm25", ["doc_a"])])
        explanation = rrf.explain(result[0], k_used=20)
        assert "doc_a" in explanation
        assert "bm25" in explanation


class TestSerialization:
    def test_to_dict_json_serializable(self):
        import json
        rrf = ReciprocalRankFusion(k=20)
        result = rrf.fuse([RankedList("bm25", ["doc_a", "doc_b"])])
        json.dumps([r.to_dict() for r in result])
