"""
Multi-Hop Query Decomposition — Week 3, System 3

THE PROBLEM
A complex research query like "Impact of AI on semiconductor manufacturing"
has several implicit sub-questions bundled into one string: supply chain
automation, chip fabrication yield improvements, labor displacement,
geopolitical export controls, and patent landscape shifts. A single flat
retrieval pass over that query string returns documents that are broadly
relevant to "AI" and "semiconductors" but frequently miss the specific
facets a thorough answer needs — the retrieval system has no way to know
the query is secretly five questions wearing one sentence.

THE FIX
Decompose the query into 3-5 targeted sub-queries BEFORE retrieval, run
them independently and in PARALLEL, then fuse the combined candidate pool
through RRF. This mirrors the coordinator-based parallel multi-query
retrieval pattern shown at SIGIR 2025's LiveRAG Challenge, where multiple
sparse/dense retrieval passes were run per query and merged via RRF before
a final reranking stage — the same architecture used here, adapted to
MARS's per-agent retrieval calls.

WHY PARALLEL, NOT SEQUENTIAL
Sequential sub-query retrieval (query 1 → results inform query 2 → ...)
is strictly slower and introduces ordering bias: whichever sub-query runs
first shapes what "counts" as relevant context for the rest. Running all
sub-queries concurrently and fusing afterward keeps them genuinely
independent — each sub-query's results are judged purely on their own
relevance to their own sub-question, and RRF's rank-based fusion combines
them without one dominating due to execution order.

WHY 3-5 SUB-QUERIES, NOT MORE
Each additional sub-query is a full retrieval round-trip (BM25 + dense +
RRF). Beyond ~5, the marginal facet coverage gained rarely justifies the
added latency and API cost — most research questions decompose into a
bounded number of genuinely distinct facets, and generating more just
produces near-duplicate rephrasings of existing sub-queries.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

from app.core.llm_client import LLMClient, TokenBudget

log = structlog.get_logger(__name__)

MIN_SUB_QUERIES = 2
MAX_SUB_QUERIES = 5
MIN_QUERY_WORDS_FOR_DECOMPOSITION = 6   # Below this, decomposition adds no value


@dataclass
class SubQuery:
    text: str
    facet: str            # Human-readable label for what this sub-query targets
    priority: int = 3      # 1 (critical) - 5 (nice-to-have)

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "facet": self.facet, "priority": self.priority}


@dataclass
class DecompositionResult:
    original_query: str
    sub_queries: list[SubQuery]
    was_decomposed: bool   # False if query was too simple to warrant decomposition
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "sub_queries": [sq.to_dict() for sq in self.sub_queries],
            "was_decomposed": self.was_decomposed,
            "reasoning": self.reasoning,
        }


class QueryDecomposerError(Exception):
    pass


class MultiHopQueryDecomposer:
    """
    Decomposes a research query into parallel, targeted sub-queries.

    Usage:
        decomposer = MultiHopQueryDecomposer(budget, session_id)
        result = await decomposer.decompose(
            "Impact of AI on semiconductor manufacturing"
        )
        # result.sub_queries -> 3-5 SubQuery objects to retrieve in parallel
    """

    def __init__(self, budget: TokenBudget, session_id: str):
        self.llm = LLMClient("planner", budget)
        self.session_id = session_id

    async def decompose(
        self,
        query: str,
        domain: str = "general",
        force: bool = False,
    ) -> DecompositionResult:
        word_count = len(query.split())

        if not force and word_count < MIN_QUERY_WORDS_FOR_DECOMPOSITION:
            log.info("query_decomposer.skipped_simple_query",
                    query=query, word_count=word_count)
            return DecompositionResult(
                original_query=query,
                sub_queries=[SubQuery(text=query, facet="direct", priority=1)],
                was_decomposed=False,
                reasoning="Query is short/simple enough to retrieve directly.",
            )

        system = """You decompose complex research queries into distinct, targeted
sub-queries for parallel retrieval. Each sub-query must target a genuinely
different facet — not a rephrasing of the same idea. Return ONLY valid JSON."""

        prompt = f"""Query: {query}
Domain: {domain}

Decompose this into 3-5 sub-queries, each targeting a distinct facet needed
to fully answer the original query. Each sub-query should be independently
retrievable — specific enough to return targeted results on its own facet.

Return JSON:
{{
  "needs_decomposition": true/false,
  "reasoning": "One sentence on why this does/doesn't need decomposition",
  "sub_queries": [
    {{"text": "specific sub-query", "facet": "what this targets", "priority": 1-5}}
  ]
}}

If the query is already narrow and singular, set needs_decomposition to
false and return a single sub-query identical to the original."""

        try:
            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=768,
                temperature=0.2,
                session_id=self.session_id,
            )
            data = json.loads(response.strip())
        except json.JSONDecodeError as exc:
            log.warning("query_decomposer.parse_failed", error=str(exc))
            return DecompositionResult(
                original_query=query,
                sub_queries=[SubQuery(text=query, facet="direct", priority=1)],
                was_decomposed=False,
                reasoning="Decomposition parse failed — falling back to direct query.",
            )
        except Exception as exc:
            log.error("query_decomposer.llm_failed", error=str(exc))
            raise QueryDecomposerError(f"Decomposition failed: {exc}") from exc

        needs_decomp = bool(data.get("needs_decomposition", True))
        raw_subs = data.get("sub_queries", [])

        if not needs_decomp or not raw_subs:
            return DecompositionResult(
                original_query=query,
                sub_queries=[SubQuery(text=query, facet="direct", priority=1)],
                was_decomposed=False,
                reasoning=data.get("reasoning", "Query judged simple enough for direct retrieval."),
            )

        sub_queries = [
            SubQuery(
                text=item.get("text", query),
                facet=item.get("facet", "unlabeled"),
                priority=int(item.get("priority", 3)),
            )
            for item in raw_subs
            if isinstance(item, dict) and item.get("text")
        ][:MAX_SUB_QUERIES]

        if len(sub_queries) < MIN_SUB_QUERIES:
            sub_queries.append(SubQuery(text=query, facet="original", priority=1))

        log.info("query_decomposer.complete",
                original=query[:60], n_sub_queries=len(sub_queries),
                facets=[sq.facet for sq in sub_queries])

        return DecompositionResult(
            original_query=query,
            sub_queries=sub_queries,
            was_decomposed=True,
            reasoning=data.get("reasoning", ""),
        )
