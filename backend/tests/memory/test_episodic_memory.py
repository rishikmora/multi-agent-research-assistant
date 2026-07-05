"""
Tests for EpisodicMemoryStore (Tier 2).
Uses an in-memory SQLite fallback for pure-Python CI, or a real
PostgreSQL test database for full integration testing (recommended,
since ARRAY and JSONB columns behave differently on SQLite).

Run: pytest tests/memory/test_episodic_memory.py -v
"""
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from app.memory.episodic import (
    Base,
    EpisodicMemoryStore,
    EpisodeWriteRequest,
    EpisodicMemoryError,
    DEFAULT_RETENTION_DAYS,
)


@pytest_asyncio.fixture
async def db_session():
    """Real PostgreSQL test DB required for ARRAY/JSONB — use a Docker
    testcontainer or a dedicated test database in CI."""
    engine = create_async_engine(
        "postgresql+asyncpg://mars_test:mars_test@localhost:5432/mars_test_db",
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture
async def store(db_session):
    return EpisodicMemoryStore(db_session)


def make_request(**overrides) -> EpisodeWriteRequest:
    defaults = dict(
        session_id=uuid4(),
        query="What are quantum computing breakthroughs in 2025?",
        query_domain="technology",
        status="complete",
        final_confidence=0.82,
        topics=["quantum hardware", "error correction"],
        settled_beliefs={"quantum hardware": {"claim": "IBM Condor exists", "confidence": 0.85}},
        contradictions_found=1,
        contradictions_resolved=1,
        source_urls=["https://arxiv.org/abs/1234", "https://ibm.com/quantum"],
        avg_source_trust=0.78,
        total_tokens=45000,
        duration_seconds=42.3,
        refinement_iterations=1,
        report_summary="Quantum computing has seen significant hardware progress.",
    )
    defaults.update(overrides)
    return EpisodeWriteRequest(**defaults)


class TestRecordEpisode:
    @pytest.mark.asyncio
    async def test_record_creates_episode(self, store):
        request = make_request()
        episode = await store.record_episode(request)

        assert episode.id is not None
        assert episode.query == request.query
        assert episode.final_confidence == 0.82

    @pytest.mark.asyncio
    async def test_record_creates_topic_index_entries(self, store, db_session):
        from sqlalchemy import select
        from app.memory.episodic import EpisodeTopicIndex

        request = make_request(topics=["topic_a", "topic_b"])
        episode = await store.record_episode(request)

        stmt = select(EpisodeTopicIndex).where(EpisodeTopicIndex.episode_id == episode.id)
        result = await db_session.execute(stmt)
        entries = result.scalars().all()

        assert len(entries) == 2
        normalized = {e.topic_normalized for e in entries}
        assert normalized == {"topic_a", "topic_b"}

    @pytest.mark.asyncio
    async def test_source_urls_are_bounded_to_100(self, store):
        request = make_request(source_urls=[f"https://example.com/{i}" for i in range(150)])
        episode = await store.record_episode(request)
        assert len(episode.source_urls) == 100
        assert episode.source_count == 150   # Count reflects the true total

    @pytest.mark.asyncio
    async def test_failed_episode_can_be_recorded(self, store):
        request = make_request(
            status="failed",
            final_confidence=None,
            error_message="LLM timeout after 3 retries",
        )
        episode = await store.record_episode(request)
        assert episode.status == "failed"
        assert episode.error_message == "LLM timeout after 3 retries"


class TestSupersession:
    @pytest.mark.asyncio
    async def test_new_episode_marks_older_matching_topic_as_superseded(self, store):
        first = await store.record_episode(
            make_request(topics=["quantum error correction"], final_confidence=0.6)
        )
        second = await store.record_episode(
            make_request(topics=["quantum error correction"], final_confidence=0.85)
        )

        refreshed_first = await store.get_by_session_id(first.session_id)
        assert refreshed_first.superseded_by == second.id

    @pytest.mark.asyncio
    async def test_non_overlapping_topics_are_not_superseded(self, store):
        first = await store.record_episode(make_request(topics=["topic_x"]))
        await store.record_episode(make_request(topics=["topic_y"]))

        refreshed_first = await store.get_by_session_id(first.session_id)
        assert refreshed_first.superseded_by is None

    @pytest.mark.asyncio
    async def test_find_by_topic_excludes_superseded_by_default(self, store):
        await store.record_episode(
            make_request(topics=["evolving_topic"], final_confidence=0.5)
        )
        await store.record_episode(
            make_request(topics=["evolving_topic"], final_confidence=0.9)
        )

        result = await store.find_by_topic("evolving_topic")
        # Only the newer, non-superseded episode should appear
        assert result.total_matching == 1
        assert result.episodes[0]["final_confidence"] == 0.9


class TestFindByTopic:
    @pytest.mark.asyncio
    async def test_exact_match(self, store):
        await store.record_episode(make_request(topics=["exact topic name"]))
        result = await store.find_by_topic("exact topic name", fuzzy=False)
        assert result.total_matching == 1

    @pytest.mark.asyncio
    async def test_fuzzy_match(self, store):
        await store.record_episode(make_request(topics=["quantum error correction methods"]))
        result = await store.find_by_topic("error correction", fuzzy=True)
        assert result.total_matching >= 1

    @pytest.mark.asyncio
    async def test_no_match_returns_empty(self, store):
        result = await store.find_by_topic("completely unrelated xyz123", fuzzy=True)
        assert result.total_matching == 0
        assert result.episodes == []

    @pytest.mark.asyncio
    async def test_filters_by_user_id(self, store):
        user_a = uuid4()
        user_b = uuid4()
        await store.record_episode(make_request(topics=["shared topic"], user_id=user_a))
        await store.record_episode(make_request(topics=["shared topic"], user_id=user_b))

        result = await store.find_by_topic("shared topic", user_id=user_a)
        assert result.total_matching == 1


class TestUserHistory:
    @pytest.mark.asyncio
    async def test_returns_user_episodes_newest_first(self, store):
        user_id = uuid4()
        await store.record_episode(make_request(user_id=user_id, query="first query"))
        await store.record_episode(make_request(user_id=user_id, query="second query"))

        history = await store.get_user_history(user_id)
        assert len(history) == 2
        assert history[0]["query"] == "second query"   # Newest first

    @pytest.mark.asyncio
    async def test_respects_limit_cap(self, store):
        from app.memory.episodic import MAX_EPISODES_PER_QUERY
        user_id = uuid4()
        for i in range(MAX_EPISODES_PER_QUERY + 10):
            await store.record_episode(make_request(user_id=user_id, query=f"query {i}"))

        history = await store.get_user_history(user_id, limit=1000)  # Request way more
        assert len(history) <= MAX_EPISODES_PER_QUERY   # Cap enforced regardless of request


class TestStats:
    @pytest.mark.asyncio
    async def test_stats_aggregate_correctly(self, store):
        await store.record_episode(make_request(final_confidence=0.8, total_tokens=1000))
        await store.record_episode(make_request(final_confidence=0.6, total_tokens=2000))

        stats = await store.get_stats()
        assert stats["total_episodes"] == 2
        assert stats["avg_confidence"] == pytest.approx(0.7, abs=0.01)
        assert stats["total_tokens_consumed"] == 3000


class TestPruning:
    @pytest.mark.asyncio
    async def test_prune_only_deletes_superseded_and_expired(self, store, db_session):
        old_superseded = await store.record_episode(
            make_request(topics=["prune_test"], final_confidence=0.5)
        )
        # Manually backdate creation for the test
        old_superseded.created_at = datetime.now(timezone.utc) - timedelta(
            days=DEFAULT_RETENTION_DAYS + 10
        )
        await db_session.commit()

        newer = await store.record_episode(
            make_request(topics=["prune_test"], final_confidence=0.9)
        )

        deleted_count = await store.prune_expired()

        assert deleted_count == 1
        remaining = await store.get_by_session_id(old_superseded.session_id)
        assert remaining is None   # Pruned
        still_there = await store.get_by_session_id(newer.session_id)
        assert still_there is not None   # Newest episode always survives

    @pytest.mark.asyncio
    async def test_non_superseded_episodes_are_never_pruned_regardless_of_age(
        self, store, db_session
    ):
        """The most recent conclusion on any topic is kept indefinitely —
        pruning only removes SUPERSEDED history, never the current
        understanding, no matter how old."""
        episode = await store.record_episode(make_request(topics=["ancient_topic"]))
        episode.created_at = datetime.now(timezone.utc) - timedelta(days=3650)  # 10 years
        await db_session.commit()

        deleted_count = await store.prune_expired()

        assert deleted_count == 0
        still_there = await store.get_by_session_id(episode.session_id)
        assert still_there is not None
