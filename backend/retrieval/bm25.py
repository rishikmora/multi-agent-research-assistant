"""
BM25 Sparse Retrieval — Week 3, System 1

WHY BM25 STILL MATTERS IN 2025-2026
Dense embeddings fail silently on exact identifiers, code snippets, product
codes, rare proper nouns, and numeric values — cases where the literal
token match matters more than semantic similarity. Production hybrid
retrieval benchmarks consistently show BM25 outperforming multi-billion-
parameter dense embedding models on a meaningful slice of real-world
queries — precisely the "what was actually said" queries that dense
retrieval's "what was meant" bias gets wrong. A research agent asking
about "IBM Condor 1121 qubits" needs the exact number 1121 to match;
a dense embedding of "IBM Condor 1121 qubits" and "IBM Condor processor
qubit count" can be highly similar in vector space while missing the
one document that has the literal figure.

THIS IMPLEMENTATION
Okapi BM25 from scratch — no Elasticsearch/OpenSearch dependency, because
our corpus per query is the candidate pool from web search (10-50
documents), not a persistent multi-million-document index. Standing up
a full search engine for that scale is unjustified operational overhead;
an in-memory scoring pass is faster to build, faster to run, and has zero
additional infrastructure to operate or secure.

Parameters k1=1.5, b=0.75 are the standard Robertson/Sparck-Jones
defaults, appropriate for short-to-medium length documents (web snippets,
paper abstracts) — the mixed-length content this system actually retrieves.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Robertson/Sparck-Jones standard defaults for short-to-medium documents
DEFAULT_K1 = 1.5   # Term frequency saturation — higher = TF matters more
DEFAULT_B = 0.75    # Length normalization strength — 0=off, 1=full

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """
    Simple, deterministic tokenizer: lowercase, alphanumeric runs, keeps
    hyphenated/underscored compounds intact (e.g. "gpt-4", "self-attention")
    since splitting those apart loses exactly the identifier-matching
    signal BM25 exists to provide.
    """
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass
class BM25Document:
    """A single document in the BM25 corpus, with precomputed term stats."""
    doc_id: str
    text: str
    tokens: list[str] = field(default_factory=list)
    term_freqs: Counter = field(default_factory=Counter)
    length: int = 0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = tokenize(self.text)
        self.term_freqs = Counter(self.tokens)
        self.length = len(self.tokens)


@dataclass
class BM25Result:
    doc_id: str
    score: float
    rank: int   # 1-indexed — this is what RRF fusion consumes, not the raw score

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "score": round(self.score, 4), "rank": self.rank}


class BM25Error(Exception):
    pass


class BM25Index:
    """
    In-memory Okapi BM25 index, built fresh per query's candidate pool.

    Usage:
        index = BM25Index(k1=1.5, b=0.75)
        index.add_documents([("doc1", "text..."), ("doc2", "text...")])
        results = index.search("query terms", top_k=10)
    """

    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B):
        if k1 < 0:
            raise BM25Error(f"k1 must be non-negative, got {k1}")
        if not (0.0 <= b <= 1.0):
            raise BM25Error(f"b must be in [0, 1], got {b}")
        self.k1 = k1
        self.b = b
        self._documents: dict[str, BM25Document] = {}
        self._doc_freq: Counter = Counter()   # How many docs contain each term
        self._avg_doc_length: float = 0.0
        self._built = False

    def add_documents(self, docs: list[tuple[str, str]]) -> None:
        """docs: list of (doc_id, text) pairs."""
        for doc_id, text in docs:
            if not text or not text.strip():
                continue
            doc = BM25Document(doc_id=doc_id, text=text)
            self._documents[doc_id] = doc
        self._rebuild_stats()

    def _rebuild_stats(self) -> None:
        if not self._documents:
            self._built = False
            return

        self._doc_freq = Counter()
        total_length = 0
        for doc in self._documents.values():
            unique_terms = set(doc.tokens)
            for term in unique_terms:
                self._doc_freq[term] += 1
            total_length += doc.length

        self._avg_doc_length = total_length / len(self._documents)
        self._built = True

    def _idf(self, term: str) -> float:
        """
        BM25's IDF variant (not classic tf-idf IDF) — includes the +1 in
        the numerator, which keeps IDF non-negative even for terms that
        appear in every document (classic IDF can go negative there,
        which would penalize common-but-relevant terms incorrectly).
        """
        n = len(self._documents)
        df = self._doc_freq.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def _score_document(self, doc: BM25Document, query_terms: list[str]) -> float:
        score = 0.0
        for term in query_terms:
            tf = doc.term_freqs.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf(term)
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (
                1 - self.b + self.b * (doc.length / self._avg_doc_length)
            )
            score += idf * (numerator / denominator)
        return score

    def search(self, query: str, top_k: int = 10) -> list[BM25Result]:
        if not self._built:
            log.warning("bm25.search_on_empty_index", query=query[:50])
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scored: list[tuple[str, float]] = []
        for doc_id, doc in self._documents.items():
            score = self._score_document(doc, query_terms)
            if score > 0:
                scored.append((doc_id, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = [
            BM25Result(doc_id=doc_id, score=score, rank=i + 1)
            for i, (doc_id, score) in enumerate(scored[:top_k])
        ]

        log.info("bm25.search_complete",
                query=query[:50], candidates=len(self._documents),
                matched=len(scored), returned=len(results))
        return results

    @property
    def document_count(self) -> int:
        return len(self._documents)

    @property
    def avg_document_length(self) -> float:
        return self._avg_doc_length

    def get_document(self, doc_id: str) -> BM25Document | None:
        return self._documents.get(doc_id)
