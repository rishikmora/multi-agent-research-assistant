"""
FActScore Hallucination Benchmark — Week 4, System 1

Grounded in: Min et al. (2023), "FActScore: Fine-grained Atomic Evaluation
of Factual Precision in Long Form Text Generation" (EMNLP 2023).

THE METHODOLOGY, EXACTLY AS THE PAPER DEFINES IT
FActScore decomposes long-form generated text into "atomic facts" —
short, self-contained, independently verifiable statements — then checks
each atomic fact for support against a reliable knowledge source. The
final score is the percentage of atomic facts judged SUPPORTED.

This is a genuinely different, stricter methodology than sentence-level
or passage-level hallucination checks: a single sentence in a MARS report
often contains 2-4 distinct factual assertions bundled together (e.g.
"IBM's Condor processor, released in late 2023, reached 1,121 qubits and
represented a major leap in error correction" contains at least three
separate checkable facts: the release timing, the qubit count, and the
error-correction claim). Sentence-level checking would mark that whole
sentence SUPPORTED if even one of the three facts is grounded, silently
hiding hallucination in the other two. Atomic decomposition prevents this.

WHAT THIS IMPLEMENTATION DOES DIFFERENTLY FROM A NAIVE PORT
The original FActScore paper verifies facts against Wikipedia passages
about a specific named entity (the paper's benchmark is biography
generation). MARS's knowledge source is instead the retrieved evidence
pool from that specific research session — the actual Source objects the
Researcher agents cited. This is the correct adaptation: FActScore's
core contribution is the decomposition-then-verify METHODOLOGY, not a
commitment to any particular knowledge source; the paper's own follow-up
work (SAFE, VeriScore) already generalizes the knowledge source to
retrieval-augmented search rather than a fixed Wikipedia corpus.

A DOCUMENTED LIMITATION, STATED HONESTLY
FActScore's decomposition-centric, fact-level approach cannot detect
manipulations that reorder or montage otherwise-true statements into a
deceptive overall narrative — each atomic fact can be individually true
while the composed passage misleads. Benchmark work (MontageLie, 2025)
shows fine-grained evaluators including FActScore can be defeated this
way, with detection AUC-ROC below 65% on such adversarial cases. This
system does not claim to catch narrative-level manipulation — it
measures atomic factual precision, which is what it was designed to do.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import structlog

from app.core.llm_client import LLMClient, TokenBudget

log = structlog.get_logger(__name__)

MAX_ATOMIC_FACTS_PER_BATCH = 20
MIN_TEXT_LENGTH_FOR_DECOMPOSITION = 20


class FactSupportVerdict(str, Enum):
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    IRRELEVANT = "irrelevant"


@dataclass
class AtomicFact:
    id: str = field(default_factory=lambda: str(uuid4()))
    text: str = ""
    source_sentence: str = ""
    verdict: FactSupportVerdict | None = None
    supporting_source_ids: list[str] = field(default_factory=list)
    verification_reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source_sentence": self.source_sentence[:200],
            "verdict": self.verdict.value if self.verdict else None,
            "supporting_source_ids": self.supporting_source_ids,
            "verification_reasoning": self.verification_reasoning,
        }


@dataclass
class FActScoreResult:
    total_atomic_facts: int = 0
    supported_facts: int = 0
    not_supported_facts: int = 0
    irrelevant_facts: int = 0
    score: float = 0.0
    facts: list[AtomicFact] = field(default_factory=list)
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def flagged_facts(self) -> list[AtomicFact]:
        return [f for f in self.facts if f.verdict == FactSupportVerdict.NOT_SUPPORTED]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_atomic_facts": self.total_atomic_facts,
            "supported_facts": self.supported_facts,
            "not_supported_facts": self.not_supported_facts,
            "irrelevant_facts": self.irrelevant_facts,
            "score": round(self.score, 4),
            "hallucination_rate": round(1.0 - self.score, 4) if self.total_atomic_facts else None,
            "flagged_facts": [f.to_dict() for f in self.flagged_facts],
            "evaluated_at": self.evaluated_at.isoformat(),
        }


class FActScoreError(Exception):
    pass


class AtomicFactDecomposer:
    """Stage 1 of FActScore: break generated text into atomic facts."""

    def __init__(self, budget: TokenBudget, session_id: str):
        self.llm = LLMClient("critic", budget)
        self.session_id = session_id

    async def decompose(self, text: str) -> list[AtomicFact]:
        if len(text.strip()) < MIN_TEXT_LENGTH_FOR_DECOMPOSITION:
            return []

        system = """You decompose text into atomic facts — short, self-contained,
independently verifiable statements. Each atomic fact must:
- Contain exactly ONE checkable claim (split compound sentences apart)
- Be understandable without needing the rest of the passage as context
- Preserve specific entities, numbers, and dates exactly as stated
Return ONLY valid JSON."""

        prompt = f"""Decompose this text into atomic facts (max {MAX_ATOMIC_FACTS_PER_BATCH}):

TEXT:
{text[:3000]}

Return JSON array:
[{{"fact": "single atomic claim", "source_sentence": "the sentence it came from"}}]

Example: "IBM's Condor, released in 2023, reached 1121 qubits and improved error correction."
Decomposes to:
[
  {{"fact": "IBM's Condor was released in 2023", "source_sentence": "IBM's Condor, released in 2023, reached 1121 qubits..."}},
  {{"fact": "IBM's Condor has 1121 qubits", "source_sentence": "..."}},
  {{"fact": "IBM's Condor improved error correction", "source_sentence": "..."}}
]"""

        try:
            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=2048,
                temperature=0.0,
                session_id=self.session_id,
            )
            raw = json.loads(response.strip())
        except json.JSONDecodeError as exc:
            log.warning("fact_decomposer.parse_failed", error=str(exc))
            return []
        except Exception as exc:
            log.error("fact_decomposer.llm_failed", error=str(exc))
            raise FActScoreError(f"Fact decomposition failed: {exc}") from exc

        facts = [
            AtomicFact(
                text=item.get("fact", ""),
                source_sentence=item.get("source_sentence", ""),
            )
            for item in raw[:MAX_ATOMIC_FACTS_PER_BATCH]
            if isinstance(item, dict) and item.get("fact")
        ]

        log.info("fact_decomposer.complete", n_facts=len(facts), text_len=len(text))
        return facts


class FactVerifier:
    """Stage 2 of FActScore: verify each atomic fact against the knowledge source."""

    def __init__(self, budget: TokenBudget, session_id: str):
        self.llm = LLMClient("critic", budget)
        self.session_id = session_id

    async def verify_batch(
        self,
        facts: list[AtomicFact],
        sources: list[dict[str, Any]],
    ) -> list[AtomicFact]:
        if not facts:
            return []
        if not sources:
            for f in facts:
                f.verdict = FactSupportVerdict.NOT_SUPPORTED
                f.verification_reasoning = "No sources available to verify against."
            return facts

        source_block = "\n".join(
            f"[{s['id']}] {s.get('title', 'Untitled')}: {s.get('snippet', '')[:250]}"
            for s in sources[:20]
        )
        facts_block = "\n".join(
            f"{i}. {f.text}" for i, f in enumerate(facts)
        )

        system = """You verify atomic facts against provided sources. For each fact,
determine if the sources SUPPORT it, do NOT support it (absent or contradicted),
or if the fact is IRRELEVANT (not the kind of claim these sources could confirm).
Be strict: paraphrase-level support counts, but unsupported inference does not.
Return ONLY valid JSON."""

        prompt = f"""SOURCES:
{source_block}

ATOMIC FACTS TO VERIFY:
{facts_block}

Return JSON array (one entry per fact, in order):
[
  {{
    "index": 0,
    "verdict": "supported|not_supported|irrelevant",
    "supporting_source_ids": ["source_id_if_supported"],
    "reasoning": "brief explanation"
  }}
]"""

        try:
            response = await self.llm.complete(
                messages=[{"role": "user", "content": prompt}],
                system=system,
                max_tokens=2048,
                temperature=0.0,
                session_id=self.session_id,
            )
            raw = json.loads(response.strip())
        except json.JSONDecodeError as exc:
            log.warning("fact_verifier.parse_failed", error=str(exc))
            for f in facts:
                f.verdict = FactSupportVerdict.NOT_SUPPORTED
                f.verification_reasoning = "Verification parse failure — treated as unsupported."
            return facts
        except Exception as exc:
            log.error("fact_verifier.llm_failed", error=str(exc))
            raise FActScoreError(f"Fact verification failed: {exc}") from exc

        verdict_map = {item.get("index"): item for item in raw if isinstance(item, dict)}

        for i, fact in enumerate(facts):
            item = verdict_map.get(i)
            if not item:
                fact.verdict = FactSupportVerdict.NOT_SUPPORTED
                fact.verification_reasoning = "No verdict returned — treated as unsupported."
                continue
            try:
                fact.verdict = FactSupportVerdict(item.get("verdict", "not_supported"))
            except ValueError:
                fact.verdict = FactSupportVerdict.NOT_SUPPORTED
            fact.supporting_source_ids = item.get("supporting_source_ids", [])
            fact.verification_reasoning = item.get("reasoning", "")

        return facts


class FActScoreBenchmark:
    """
    Full FActScore pipeline: decompose -> verify -> score.

    Usage:
        benchmark = FActScoreBenchmark(budget, session_id)
        result = await benchmark.evaluate(report_text, sources)
        result.score            # 0-1, % of non-irrelevant facts supported
        result.flagged_facts    # Specific unsupported claims for review
    """

    def __init__(self, budget: TokenBudget, session_id: str):
        self.decomposer = AtomicFactDecomposer(budget, session_id)
        self.verifier = FactVerifier(budget, session_id)

    async def evaluate(
        self, text: str, sources: list[dict[str, Any]]
    ) -> FActScoreResult:
        facts = await self.decomposer.decompose(text)
        if not facts:
            return FActScoreResult()

        verified_facts = await self.verifier.verify_batch(facts, sources)

        supported = sum(1 for f in verified_facts if f.verdict == FactSupportVerdict.SUPPORTED)
        not_supported = sum(1 for f in verified_facts if f.verdict == FactSupportVerdict.NOT_SUPPORTED)
        irrelevant = sum(1 for f in verified_facts if f.verdict == FactSupportVerdict.IRRELEVANT)

        checkable = supported + not_supported
        score = supported / checkable if checkable > 0 else 1.0

        result = FActScoreResult(
            total_atomic_facts=len(verified_facts),
            supported_facts=supported,
            not_supported_facts=not_supported,
            irrelevant_facts=irrelevant,
            score=round(score, 4),
            facts=verified_facts,
        )

        log.info("factscore.evaluation_complete",
                total_facts=result.total_atomic_facts,
                score=result.score,
                flagged=len(result.flagged_facts))

        return result
