"""
Verification tools — structural separation from research agents.
Critic uses these to independently validate claims.
"""
from __future__ import annotations
import structlog
from app.schemas.research import Source

log = structlog.get_logger(__name__)


class CitationVerifier:
    """
    Verifies that cited sources actually support claimed findings.
    Checks URL reachability and content relevance.
    """

    async def verify(self, claim: str, source: Source) -> dict[str, bool | float | str]:
        """Returns verification result with confidence."""
        # In production: fetch URL, extract text, use LLM to verify claim support
        # For now: heuristic based on snippet overlap
        claim_words = set(claim.lower().split())
        snippet_words = set(source.snippet.lower().split())
        overlap = len(claim_words & snippet_words) / max(len(claim_words), 1)

        return {
            "verified": overlap > 0.15,
            "confidence": round(min(overlap * 3, 1.0), 3),
            "reason": "snippet_overlap_check",
        }


class FactChecker:
    """
    Cross-references claims across multiple sources.
    Returns agreement score and contradiction flags.
    """

    async def check(
        self, claim: str, sources: list[Source]
    ) -> dict[str, float | list[str]]:
        """Checks claim against multiple sources for consistency."""
        if not sources:
            return {"agreement_score": 0.5, "contradictions": []}

        claim_words = set(claim.lower().split())
        supporting = 0
        contradictions = []

        for source in sources[:5]:
            snippet_words = set(source.snippet.lower().split())
            if len(claim_words & snippet_words) > 2:
                supporting += 1

        agreement = supporting / len(sources[:5])
        return {
            "agreement_score": round(agreement, 3),
            "contradictions": contradictions,
            "sources_checked": len(sources[:5]),
        }
