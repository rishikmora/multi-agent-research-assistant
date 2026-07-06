"""
Reciprocal Rank Fusion (RRF) — Week 3, System 2

WHY RRF, NOT WEIGHTED SCORE AVERAGING
BM25 scores are typically in the 0-20+ range with an unbounded, corpus-
dependent distribution. Cosine similarity scores are bounded [-1, 1] or
[0, 1] depending on normalization. These are fundamentally incompatible
scales — averaging them directly means comparing apples to bananas, and
whichever scorer happens to produce larger raw numbers silently dominates
the fusion regardless of actual relevance.

RRF sidesteps this entirely by discarding scores and fusing on RANK
POSITION only: RRF(d) = Σ 1/(k + rank_i(d)) across all ranked lists i
that contain document d. A document ranked #1 by BM25 and #2 by dense
search accumulates 1/(k+1) + 1/(k+2) — no normalization step required,
and the method requires zero training or tuning to outperform naive
weighted score combination.

THE k PARAMETER — CORPUS-SIZE DEPENDENT, NOT A UNIVERSAL CONSTANT
k=60 is the standard default, but it is calibrated for TREC-scale corpora
with thousands of candidate documents. Our per-query candidate pool is
10-50 web/academic results — an order of magnitude smaller. At small
corpus sizes, k=60 over-flattens the rank differences that actually carry
signal: the gap between rank 1 and rank 5 out of 20 documents is far more
meaningful than the gap between rank 1 and rank 5 out of 5,000. Debugging
reports from production RAG deployments on small corpora (80-300 documents)
found k=60 produced worse-than-dense-only results, resolved by dropping to
k=10-20. This implementation defaults to k=20 for exactly that reason and
exposes it as a tunable parameter, rather than blindly inheriting the
TREC-scale default.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# Calibrated for small candidate pools (10-50 docs), not TREC-scale corpora.
# See module docstring for the reasoning — k=60 is the wrong default here.
DEFAULT_RRF_K_SMALL_CORPUS = 20
CORPUS_SIZE_THRESHOLD_FOR_LARGE_K = 500   # Above this, k=60 becomes appropriate


@dataclass
class RankedList:
    """One ranker's output — a name (for provenance) and ordered doc_ids."""
    ranker_name: str
    ranked_doc_ids: list[str]   # Already sorted best-first
    weight: float = 1.0          # Per-ranker weight, default equal trust


@dataclass
class FusedResult:
    doc_id: str
    rrf_score: float
    contributing_rankers: list[str] = field(default_factory=list)
    per_ranker_ranks: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "rrf_score": round(self.rrf_score, 6),
            "contributing_rankers": self.contributing_rankers,
            "per_ranker_ranks": self.per_ranker_ranks,
        }


class RRFFusionError(Exception):
    pass


class ReciprocalRankFusion:
    """
    Fuses N ranked lists into one, using rank-based scoring.

    Usage:
        rrf = ReciprocalRankFusion(k=20)
        fused = rrf.fuse([
            RankedList("bm25", ["doc3", "doc1", "doc7"]),
            RankedList("dense", ["doc1", "doc9", "doc3"]),
        ])
        # fused[0] is the doc with the highest combined rank-based score
    """

    def __init__(self, k: int | None = None):
        # k=None triggers corpus-size-aware auto-selection at fuse() time
        self._k_override = k

    def _select_k(self, ranked_lists: list[RankedList]) -> int:
        if self._k_override is not None:
            return self._k_override

        max_candidates = max((len(rl.ranked_doc_ids) for rl in ranked_lists), default=0)
        if max_candidates >= CORPUS_SIZE_THRESHOLD_FOR_LARGE_K:
            return 60   # TREC-scale default becomes appropriate here
        return DEFAULT_RRF_K_SMALL_CORPUS

    def fuse(self, ranked_lists: list[RankedList]) -> list[FusedResult]:
        if not ranked_lists:
            return []

        k = self._select_k(ranked_lists)

        scores: dict[str, float] = defaultdict(float)
        contributors: dict[str, list[str]] = defaultdict(list)
        rank_positions: dict[str, dict[str, int]] = defaultdict(dict)

        for ranked_list in ranked_lists:
            if ranked_list.weight <= 0:
                log.warning("rrf.non_positive_weight_skipped",
                          ranker=ranked_list.ranker_name, weight=ranked_list.weight)
                continue

            for position, doc_id in enumerate(ranked_list.ranked_doc_ids, start=1):
                contribution = ranked_list.weight / (k + position)
                scores[doc_id] += contribution
                contributors[doc_id].append(ranked_list.ranker_name)
                rank_positions[doc_id][ranked_list.ranker_name] = position

        fused = [
            FusedResult(
                doc_id=doc_id,
                rrf_score=score,
                contributing_rankers=contributors[doc_id],
                per_ranker_ranks=rank_positions[doc_id],
            )
            for doc_id, score in scores.items()
        ]
        fused.sort(key=lambda r: r.rrf_score, reverse=True)

        log.info("rrf.fusion_complete",
                k_used=k, n_rankers=len(ranked_lists),
                n_unique_docs=len(fused),
                docs_in_all_lists=sum(
                    1 for r in fused if len(set(r.contributing_rankers)) == len(ranked_lists)
                ))
        return fused

    def explain(self, result: FusedResult, k_used: int) -> str:
        """Human-readable breakdown of how a document's score was computed."""
        terms = [
            f"{ranker}@rank{rank} → 1/({k_used}+{rank})={1/(k_used+rank):.4f}"
            for ranker, rank in result.per_ranker_ranks.items()
        ]
        return f"{result.doc_id}: " + " + ".join(terms) + f" = {result.rrf_score:.4f}"
