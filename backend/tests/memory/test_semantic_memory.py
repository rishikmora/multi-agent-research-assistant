"""
Tests for SemanticMemoryStore (Tier 3).
Requires PostgreSQL + pgvector extension in the test database.

Run: pytest tests/memory/test_semantic_memory.py -v
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.memory.semantic import (
    Base,
    SemanticMemoryStore,
    ConsolidationCandidate,
    _content_hash,
    _decay_confidence,
    CONFIDENCE_DECAY_HALF_LIFE_DAYS,
)


@pytest_asyncio.fixture
async def db_session():
    engine = create_async_engine(
        "postgresql+asyncpg://mars_test:mars_test@localhost:5432/mars_test_db",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


def fake_embedding(seed: int = 0) -> list[float]:
    """Deterministic fake 384-dim embedding for tests that don't need
    a real sentence-transformers model loaded."""
    import random
    rng = random.Random(seed)
    vec = [rng.uniform(-1, 1) for _ in range(384)]
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec]


@pytest_asyncio.fixture
async def store(db_session):
    s = SemanticMemoryStore(db_session)
    # Patch embed() to avoid loading the real model in unit tests —
    # integration tests separately verify the real encoder.
    s.embed = lambda text_input: fake_embedding(hash(text_input) % 10000)
    return s


class TestContentHash:
    def test_identical_content_same_hash(self):
        assert _content_hash("The sky is blue") == _content_hash("The sky is blue")

    def test_normalization_ignores_case_and_whitespace(self):
        assert _content_hash("Hello   World") == _content_hash("hello world")
        assert _content_hash("  Hello World  ") == _content_hash("Hello World")

    def test_different_content_different_hash(self):
        assert _content_hash("Claim A") != _content_hash("Claim B")


class TestConfidenceDecay:
    def test_no_decay_for_just_reinforced(self):
        now = datetime.now(timezone.utc)
        result = _decay_confidence(0.8, now)
        assert result == pytest.approx(0.8, abs=0.01)

    def test_half_life_produces_half_confidence(self):
        stale = datetime.now(timezone.utc) - timedelta(days=CONFIDENCE_DECAY_HALF_LIFE_DAYS)
        result = _decay_confidence(0.8, stale)
        assert result == pytest.approx(0.4, abs=0.02)

    def test_decay_floors_at_15_percent(self):
        very_stale = datetime.now(timezone.utc) - timedelta(days=CONFIDENCE_DECAY_HALF_LIFE_DAYS * 20)
        result = _decay_confidence(0.8, very_stale)
        assert result >= 0.8 * 0.15 - 0.001

    def test_naive_datetime_handled_gracefully(self):
        """Timestamps loaded from some DB drivers may lack tzinfo — must
        not crash."""
        naive = datetime.now() - timedelta(days=10)
        result = _decay_confidence(0.8, naive)
        assert 0 < result <= 0.8


class TestConsolidationCreate:
    @pytest.mark.asyncio
    async def test_new_fact_creates_entry(self, store):
        candidate = ConsolidationCandidate(
            episode_id=uuid4(),
            content="IBM's Condor processor has 1121 qubits",
            embedding=fake_embedding(1),
            topics=["quantum hardware"],
            domain="technology",
            confidence=0.85,
        )

        entry, action = await store.consolidate(candidate)

        assert action == "created"
        assert entry.content == candidate.content
        assert entry.corroboration_count == 1
        assert entry.confidence == 0.85

    @pytest.mark.asyncio
    async def test_exact_duplicate_reinforces_not_duplicates(self, store):
        candidate = ConsolidationCandidate(
            episode_id=uuid4(),
            content="The same exact claim text",
            embedding=fake_embedding(2),
            topics=["topic_a"],
            domain="general",
            confidence=0.7,
        )
        first_entry, first_action = await store.consolidate(candidate)

        second_candidate = ConsolidationCandidate(
            episode_id=uuid4(),
            content="The same exact claim text",   # Identical after normalization
            embedding=fake_embedding(2),
            topics=["topic_a"],
            domain="general",
            confidence=0.7,
        )
        second_entry, second_action = await store.consolidate(second_candidate)

        assert first_action == "created"
        assert second_action == "reinforced"
        assert second_entry.id == first_entry.id
        assert second_entry.corroboration_count == 2


class TestConsolidationReinforcement:
    @pytest.mark.asyncio
    async def test_reinforcement_increases_confidence_with_diminishing_returns(self, store):
        candidate = ConsolidationCandidate(
            episode_id=uuid4(),
            content="A fact that gets corroborated many times",
            embedding=fake_embedding(3),
            topics=[],
            domain="general",
            confidence=0.5,
        )
        entry, _ = await store.consolidate(candidate)
        first_boost_confidence = entry.confidence

        for i in range(5):
            reinforcement = ConsolidationCandidate(
                episode_id=uuid4(),
                content="A fact that gets corroborated many times",
                embedding=fake_embedding(3),
                topics=[],
                domain="general",
                confidence=0.5,
            )
            entry, action = await store.consolidate(reinforcement)
            assert action == "reinforced"

        assert entry.confidence > first_boost_confidence
        assert entry.confidence <= 0.97   # Ceiling never exceeded
        assert entry.corroboration_count == 6

    @pytest.mark.asyncio
    async def test_reinforcement_tracks_source_episodes(self, store):
        ep1, ep2 = uuid4(), uuid4()
        content = "Tracked source fact"

        await store.consolidate(ConsolidationCandidate(
            episode_id=ep1, content=content, embedding=fake_embedding(4),
            topics=[], domain="general", confidence=0.6,
        ))
        entry, _ = await store.consolidate(ConsolidationCandidate(
            episode_id=ep2, content=content, embedding=fake_embedding(4),
            topics=[], domain="general", confidence=0.6,
        ))

        assert ep1 in entry.source_episode_ids
        assert ep2 in entry.source_episode_ids

    @pytest.mark.asyncio
    async def test_resets_staleness_clock_on_reinforcement(self, store):
        candidate = ConsolidationCandidate(
            episode_id=uuid4(), content="Freshness test fact", embedding=fake_embedding(5),
            topics=[], domain="general", confidence=0.6,
        )
        entry, _ = await store.consolidate(candidate)
        entry.last_reinforced_at = datetime.now(timezone.utc) - timedelta(days=60)
        await store.db.commit()

        await store.consolidate(ConsolidationCandidate(
            episode_id=uuid4(), content="Freshness test fact", embedding=fake_embedding(5),
            topics=[], domain="general", confidence=0.6,
        ))

        recency = (datetime.now(timezone.utc) - entry.last_reinforced_at).total_seconds()
        assert recency < 5   # Just refreshed, not 60 days stale anymore


class TestContradictionDetection:
    @pytest.mark.asyncio
    async def test_contradicting_claim_flags_conflict_not_overwrite(self, store):
        original = ConsolidationCandidate(
            episode_id=uuid4(),
            content="Fault-tolerant quantum computing will arrive by 2028",
            embedding=fake_embedding(6),
            topics=["quantum timeline"],
            domain="technology",
            confidence=0.6,
        )
        entry, action = await store.consolidate(original)
        assert action == "created"

        with patch.object(store, "_check_contradiction", new=AsyncMock(return_value=True)):
            with patch.object(store, "search", new=AsyncMock(return_value=[
                type("R", (), {"entry": entry.to_dict(), "similarity": 0.9})()
            ])):
                conflicting = ConsolidationCandidate(
                    episode_id=uuid4(),
                    content="Fault-tolerant quantum computing will NOT arrive before 2035",
                    embedding=fake_embedding(6),
                    topics=["quantum timeline"],
                    domain="technology",
                    confidence=0.65,
                )
                result_entry, action = await store.consolidate(conflicting)

        assert action == "conflict"
        assert result_entry.is_contested is True
        # Original content must NOT have been silently overwritten
        assert result_entry.content == "Fault-tolerant quantum computing will arrive by 2028"

    @pytest.mark.asyncio
    async def test_llm_failure_during_contradiction_check_fails_safe(self, store):
        """If the LLM call to check contradiction fails, consolidation
        must not crash — it should fail safe (treat as non-contradiction)
        rather than blocking the whole background consolidation job."""
        with patch.object(
            store, "_check_contradiction",
            new=AsyncMock(side_effect=Exception("LLM timeout"))
        ):
            original = ConsolidationCandidate(
                episode_id=uuid4(), content="Some claim", embedding=fake_embedding(7),
                topics=[], domain="general", confidence=0.6,
            )
            await store.consolidate(original)

            with patch.object(store, "search", new=AsyncMock(return_value=[
                type("R", (), {"entry": {"id": str(uuid4())}, "similarity": 0.9})()
            ])):
                # Should not raise — _check_contradiction catches internally
                is_contra = await store._check_contradiction("A", "B")
                assert is_contra is False


class TestConflictResolution:
    @pytest.mark.asyncio
    async def test_resolve_conflict_marks_resolved_and_updates_content(self, store):
        original = ConsolidationCandidate(
            episode_id=uuid4(), content="Original disputed claim", embedding=fake_embedding(8),
            topics=[], domain="general", confidence=0.6,
        )
        entry, _ = await store.consolidate(original)
        entry.is_contested = True
        await store.db.commit()

        await store._flag_conflict(entry, ConsolidationCandidate(
            episode_id=uuid4(), content="Conflicting claim", embedding=fake_embedding(8),
            topics=[], domain="general", confidence=0.6,
        ))

        conflicts = await store.get_pending_conflicts()
        assert len(conflicts) >= 1

        conflict_id = conflicts[0]["id"]
        from uuid import UUID as _UUID
        await store.resolve_conflict(
            _UUID(conflict_id),
            resolution="Verified via primary source — original claim was correct",
            winning_claim="Original disputed claim (verified)",
        )

        pending_after = await store.get_pending_conflicts()
        assert len(pending_after) == len(conflicts) - 1


class TestSearch:
    @pytest.mark.asyncio
    async def test_search_respects_similarity_threshold(self, store):
        await store.consolidate(ConsolidationCandidate(
            episode_id=uuid4(), content="Specific unique fact XYZ123", embedding=fake_embedding(9),
            topics=[], domain="general", confidence=0.7,
        ))

        results = await store.search(
            "Specific unique fact XYZ123", similarity_threshold=0.99
        )
        # Same text embedded via the same fake function → identical vector → similarity ~1.0
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_search_filters_by_domain(self, store):
        await store.consolidate(ConsolidationCandidate(
            episode_id=uuid4(), content="Tech domain fact", embedding=fake_embedding(10),
            topics=[], domain="technology", confidence=0.7,
        ))
        await store.consolidate(ConsolidationCandidate(
            episode_id=uuid4(), content="Medical domain fact", embedding=fake_embedding(11),
            topics=[], domain="medical", confidence=0.7,
        ))

        results = await store.search("fact", domain="technology", similarity_threshold=0.0)
        domains = {r.entry["domain"] for r in results}
        assert domains <= {"technology"}

    @pytest.mark.asyncio
    async def test_format_for_agent_context_empty_when_no_results(self, store):
        context = await store.format_for_agent_context(
            "completely novel query with no history xyz999"
        )
        assert context == ""

    @pytest.mark.asyncio
    async def test_format_for_agent_context_flags_contested_entries(self, store):
        candidate = ConsolidationCandidate(
            episode_id=uuid4(), content="A contested fact for formatting test",
            embedding=fake_embedding(12), topics=[], domain="general", confidence=0.7,
        )
        entry, _ = await store.consolidate(candidate)
        entry.is_contested = True
        await store.db.commit()

        context = await store.format_for_agent_context(
            "A contested fact for formatting test", top_k=1
        )
        assert "[CONTESTED]" in context


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_reflect_stored_entries(self, store):
        await store.consolidate(ConsolidationCandidate(
            episode_id=uuid4(), content="Stat fact 1", embedding=fake_embedding(13),
            topics=[], domain="general", confidence=0.8,
        ))
        await store.consolidate(ConsolidationCandidate(
            episode_id=uuid4(), content="Stat fact 2", embedding=fake_embedding(14),
            topics=[], domain="general", confidence=0.6,
        ))

        stats = await store.get_stats()
        assert stats["total_entries"] == 2
        assert stats["avg_confidence"] == pytest.approx(0.7, abs=0.01)
