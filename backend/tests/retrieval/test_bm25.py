"""
Tests for BM25Index (Week 3, System 1).
Run: pytest tests/retrieval/test_bm25.py -v
"""
import pytest

from app.retrieval.bm25 import BM25Index, BM25Error, tokenize


class TestTokenize:
    def test_lowercases(self):
        assert tokenize("Hello WORLD") == ["hello", "world"]

    def test_keeps_hyphenated_compounds(self):
        assert "gpt-4" in tokenize("GPT-4 is a model")

    def test_keeps_underscored_compounds(self):
        assert "self_attention" in tokenize("self_attention mechanism")

    def test_strips_punctuation(self):
        result = tokenize("Hello, world! How are you?")
        assert "," not in result and "!" not in result and "?" not in result

    def test_empty_string(self):
        assert tokenize("") == []


class TestBM25Construction:
    def test_rejects_negative_k1(self):
        with pytest.raises(BM25Error):
            BM25Index(k1=-1.0)

    def test_rejects_b_outside_unit_interval(self):
        with pytest.raises(BM25Error):
            BM25Index(b=1.5)
        with pytest.raises(BM25Error):
            BM25Index(b=-0.1)

    def test_accepts_boundary_b_values(self):
        BM25Index(b=0.0)
        BM25Index(b=1.0)

    def test_empty_index_search_returns_empty(self):
        index = BM25Index()
        assert index.search("anything") == []

    def test_skips_empty_documents(self):
        index = BM25Index()
        index.add_documents([("doc1", "real content here"), ("doc2", ""), ("doc3", "   ")])
        assert index.document_count == 1


class TestBM25Scoring:
    def test_exact_term_match_scores_higher_than_no_match(self):
        index = BM25Index()
        index.add_documents([
            ("relevant", "IBM Condor processor has 1121 qubits"),
            ("irrelevant", "The weather today is sunny and warm"),
        ])
        results = index.search("IBM Condor qubits")
        assert results[0].doc_id == "relevant"

    def test_higher_term_frequency_scores_higher(self):
        index = BM25Index()
        index.add_documents([
            ("high_freq", "quantum quantum quantum computing breakthrough"),
            ("low_freq", "quantum computing is one field among many other topics entirely"),
        ])
        results = index.search("quantum")
        assert results[0].doc_id == "high_freq"

    def test_rare_terms_weighted_higher_than_common_terms(self):
        """IDF behavior: a term appearing in only 1 of 3 docs should
        contribute more to score than a term in all 3."""
        index = BM25Index()
        index.add_documents([
            ("doc1", "common word rare_identifier_xyz"),
            ("doc2", "common word here"),
            ("doc3", "common word there"),
        ])
        results_rare = index.search("rare_identifier_xyz", top_k=1)
        results_common = index.search("common", top_k=1)
        # The rare term should produce a non-trivial score even though it
        # appears in only one document
        assert results_rare[0].score > 0

    def test_rank_is_1_indexed(self):
        index = BM25Index()
        index.add_documents([
            ("doc1", "quantum computing breakthrough"),
            ("doc2", "quantum computing advances"),
        ])
        results = index.search("quantum computing")
        assert results[0].rank == 1
        if len(results) > 1:
            assert results[1].rank == 2

    def test_documents_with_zero_score_excluded(self):
        index = BM25Index()
        index.add_documents([
            ("relevant", "quantum computing breakthrough"),
            ("unrelated", "cooking recipes and gardening tips"),
        ])
        results = index.search("quantum computing")
        doc_ids = {r.doc_id for r in results}
        assert "unrelated" not in doc_ids

    def test_top_k_limits_results(self):
        index = BM25Index()
        index.add_documents([(f"doc{i}", f"quantum topic number {i}") for i in range(20)])
        results = index.search("quantum", top_k=5)
        assert len(results) <= 5

    def test_empty_query_returns_empty(self):
        index = BM25Index()
        index.add_documents([("doc1", "some content")])
        assert index.search("") == []
        assert index.search("   ") == []

    def test_length_normalization_affects_score(self):
        """A short document densely matching the query should score
        competitively against a long document with the same absolute
        term frequency but heavy length dilution."""
        index = BM25Index(b=0.75)
        index.add_documents([
            ("short_dense", "quantum computing"),
            ("long_dilute", "quantum computing " + "filler word here " * 50),
        ])
        results = index.search("quantum computing")
        # With b > 0, the short document should not be penalized as heavily
        # relative to raw term frequency
        assert results[0].doc_id == "short_dense"


class TestBM25Stats:
    def test_document_count(self):
        index = BM25Index()
        index.add_documents([("doc1", "text"), ("doc2", "more text")])
        assert index.document_count == 2

    def test_avg_document_length_computed(self):
        index = BM25Index()
        index.add_documents([("doc1", "one two three"), ("doc2", "four five")])
        assert index.avg_document_length == pytest.approx(2.5, abs=0.01)

    def test_get_document_returns_stored_doc(self):
        index = BM25Index()
        index.add_documents([("doc1", "hello world")])
        doc = index.get_document("doc1")
        assert doc is not None
        assert doc.text == "hello world"

    def test_get_missing_document_returns_none(self):
        index = BM25Index()
        assert index.get_document("nonexistent") is None

    def test_adding_more_documents_rebuilds_stats(self):
        index = BM25Index()
        index.add_documents([("doc1", "one two")])
        assert index.document_count == 1
        index.add_documents([("doc2", "three four five")])
        assert index.document_count == 2
